"""
DS-41 — Sahte Haber Tespiti (Bag of Words + Naive Bayes)
Bir medya platformunda data scientist'sin. Yayınlanan haberleri sahte/yanıltıcı
(FAKE) ve gerçek (REAL) olarak ayıran bir model kuruyorsun. Bunun için
"Bag of Words" yaklaşımını (CountVectorizer — kelime SAYIMI) + MultinomialNB
kullanacaksın. Bag of Words metin sınıflandırmanın klasik, hızlı ve güçlü
baseline'ıdır.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""
import os

import numpy as np
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

NEWS_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"

# 1. fake_or_real_news.csv veri setini indir (cache'li)
def fetch_news_data(cache_dir='data'):
    """
    GitHub'dan fake_or_real_news.csv indir, path'ini döndür.

    Akış:
    1. cache_dir mevcut değilse oluştur (os.makedirs(cache_dir, exist_ok=True))
    2. Hedef dosya path: cache_dir/fake_or_real_news.csv
    3. Bu dosya zaten varsa → hemen path'i döndür (cache hit, tekrar indirme)
    4. Yoksa URL'den indir ve cache_dir'a yaz:
       https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv

    Args:
        cache_dir: lokal cache klasörü (default 'data')

    Returns:
        str: fake_or_real_news.csv dosyasının tam yolu

    İpucu: import os, requests
    - Dosya ~30MB olduğu için timeout uzun tut:
        r = requests.get(URL, timeout=120); r.raise_for_status()
        with open(data_path, 'wb') as f: f.write(r.content)
    URL: https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_path = os.path.join(cache_dir, 'fake_or_real_news.csv')

    # Cache hit
    if os.path.exists(data_path):
        return data_path

    # CSV indir (~30MB — timeout uzun tut)
    r = requests.get(NEWS_URL, timeout=120)
    r.raise_for_status()
    with open(data_path, 'wb') as f:
        f.write(r.content)

    return data_path


# 2. CSV'yi DataFrame olarak yükle
def load_news_data(path):
    """
    fake_or_real_news.csv dosyasını oku. Dosyanın İLK sütunu isimsiz bir id
    (index) sütunudur — onu AT. Sadece şu üç sütunu döndür:
    ['title', 'text', 'label'].

    Args:
        path: CSV dosya yolu (fetch_news_data'dan dönen)

    Returns:
        pd.DataFrame: ~6335 satır × 3 sütun ['title', 'text', 'label']

    İpucu:
    - df = pd.read_csv(path)
    - İlk sütun (genelde 'Unnamed: 0') bir id → df.drop(columns=[df.columns[0]])
      veya read_csv'de index_col=0 / usecols kullan
    - Son olarak df[['title', 'text', 'label']] döndür
    """
    # İlk sütun isimsiz bir id (index) — onu atla, sadece title/text/label al
    df = pd.read_csv(path)
    # İlk sütun (Unnamed: 0) bir id; varsa düşür
    first_col = df.columns[0]
    if first_col.lower().startswith('unnamed') or first_col == '':
        df = df.drop(columns=[first_col])
    return df[['title', 'text', 'label']]


# 3. Veriyi keşfet — label dağılımı
def explore_data(df):
    """
    Temel keşif metriği üret.

    Returns:
        dict: {
            'total': int (toplam haber sayısı),
            'real_count': int ('REAL' = gerçek haber sayısı),
            'fake_count': int ('FAKE' = sahte haber sayısı),
            'fake_rate': float (fake_count / total — ~0.50, dengeli)
        }

    İpucu:
    - df['label'].value_counts() → 'REAL' ve 'FAKE' sayıları
    - total = len(df)
    """
    counts = df['label'].value_counts().to_dict()
    total = len(df)
    real = int(counts.get('REAL', 0))
    fake = int(counts.get('FAKE', 0))
    return {
        'total': total,
        'real_count': real,
        'fake_count': fake,
        'fake_rate': fake / total,
    }


# 4. Label'ları encode et (FAKE/REAL → 1/0)
def encode_labels(df):
    """
    'label' sütununu binary target'a çevir: 'FAKE' → 1, 'REAL' → 0.
    Yeni bir 'target' sütununa yaz (orijinal 'label' kalsın).

    Args:
        df: 'label' sütunu olan DataFrame

    Returns:
        pd.DataFrame: yeni 'target' sütunu (int 0/1) eklenmiş kopya

    İpucu:
    - out = df.copy()
    - out['target'] = out['label'].map({'FAKE': 1, 'REAL': 0}).astype(int)
    """
    out = df.copy()
    out['target'] = out['label'].map({'FAKE': 1, 'REAL': 0}).astype(int)
    return out


# 5. Train/test split (stratified)
def split_data(X, y):
    """
    train_test_split kullan:
    - X: haber metinleri (pd.Series — ham string'ler, vektörize EDİLMEMİŞ)
    - y: target (0/1)
    - test_size=0.2
    - stratify=y (FAKE/REAL oranı train ve test'te korunsun)
    - random_state=42 (tekrarlanabilirlik)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    İpucu: from sklearn.model_selection import train_test_split
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 6. Bag of Words (CountVectorizer) + MultinomialNB pipeline kur
def build_bow_pipeline():
    """
    sklearn Pipeline — BAG OF WORDS (kelime sayımı, TF-IDF DEĞİL):
    - 'cv': CountVectorizer(stop_words='english', max_features=10000)
            → metni "bag of words" SAYIM vektörüne çevirir (her kelimenin
              dokümanda kaç kez geçtiğini sayar)
            → stop_words='english' → 'the', 'a', 'is' gibi anlamsız İngilizce
              kelimeleri eler
            → max_features=10000 → en sık geçen 10000 kelimeyi tut (sözlüğü
              sınırla, hız + bellek)
    - 'nb': MultinomialNB()
            → kelime sayımları için klasik Naive Bayes (metin sınıflandırmanın
              standart seçimi)

    Returns:
        sklearn.pipeline.Pipeline

    İpucu:
    - from sklearn.feature_extraction.text import CountVectorizer
    - from sklearn.naive_bayes import MultinomialNB
    - from sklearn.pipeline import Pipeline
    - Pipeline ham string listesini alır, CountVectorizer otomatik vektörize eder.
    - NOT: Burada CountVectorizer kullan (bag of words). TfidfVectorizer DEĞİL.
    """
    return Pipeline([
        ('cv', CountVectorizer(stop_words='english', max_features=10000)),
        ('nb', MultinomialNB()),
    ])


# 7. Modeli eğit
def train_model(pipe, X_train, y_train):
    """
    Pipeline'ı fit et ve döndür.

    Args:
        pipe: build_bow_pipeline'dan dönen pipeline
        X_train: haber metinleri (ham string'ler)
        y_train: target (0/1)

    Returns:
        Pipeline: fit edilmiş pipeline

    İpucu: pipe.fit(X_train, y_train); return pipe
    """
    pipe.fit(X_train, y_train)
    return pipe


# 8. Modeli değerlendir
def evaluate_model(pipe, X_test, y_test):
    """
    Test setinde tahmin yap, metrikleri hesapla.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.array (2x2)
        }

    İpucu:
    - y_pred = pipe.predict(X_test)
    - sklearn.metrics'ten: accuracy_score, precision_score, recall_score,
      f1_score, confusion_matrix
    - precision/recall/f1'de zero_division=0 kullan
    """
    y_pred = pipe.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


# 9. Tek bir haber için tahmin yap
def predict_article(pipe, text):
    """
    Bir haber metni al, FAKE/REAL tahmini + fake olasılığı döndür.

    Args:
        pipe: eğitilmiş pipeline
        text: str (bir haber metni)

    Returns:
        dict: {
            'prediction': int (0=REAL, 1=FAKE),
            'label': str ('FAKE' veya 'REAL'),
            'fake_probability': float (fake olma olasılığı)
        }

    İpucu:
    - Pipeline tek string değil, liste bekler → [text] geç
    - proba = pipe.predict_proba([text])[0, 1]  # pozitif (FAKE) sınıf olasılığı
    - pred = int(pipe.predict([text])[0])
    """
    proba = float(pipe.predict_proba([text])[0, 1])
    pred = int(pipe.predict([text])[0])
    return {
        'prediction': pred,
        'label': 'FAKE' if pred == 1 else 'REAL',
        'fake_probability': proba,
    }


# 10. En "fake" kelimeler
def top_fake_words(pipe, n=10):
    """
    Modelin sahte haberi en çok ayırt eden n kelimesini bul.

    MultinomialNB.feature_log_prob_ → her sınıf için her kelimenin log
    olasılığı. feature_log_prob_[1] = FAKE sınıfı, feature_log_prob_[0] = REAL.
    İkisinin FARKI en büyük olan kelimeler → "en fake" kelimeler.

    Args:
        pipe: eğitilmiş pipeline
        n: kaç kelime döndürülecek (default 10)

    Returns:
        list: n adet kelime (str), en fake'ten aza doğru sıralı

    İpucu:
    - cv = pipe.named_steps['cv']; nb = pipe.named_steps['nb']
    - feature_names = cv.get_feature_names_out()
    - diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    - top_idx = np.argsort(diff)[::-1][:n]  # büyükten küçüğe ilk n
    - [feature_names[i] for i in top_idx]
    """
    cv = pipe.named_steps['cv']
    nb = pipe.named_steps['nb']
    feature_names = cv.get_feature_names_out()
    # fake log-prob - real log-prob → fake'i en çok ayırt eden kelimeler
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top_idx = np.argsort(diff)[::-1][:n]
    return [feature_names[i] for i in top_idx]


# 11. En "real" kelimeler
def top_real_words(pipe, n=10):
    """
    top_fake_words'ün tersi — gerçek (REAL) haberleri en çok ayırt eden n kelime.

    Args:
        pipe: eğitilmiş pipeline
        n: kaç kelime (default 10)

    Returns:
        list: n adet kelime (str), en real'den aza doğru sıralı

    İpucu:
    - diff = nb.feature_log_prob_[0] - nb.feature_log_prob_[1]  # REAL - FAKE
    - top_idx = np.argsort(diff)[::-1][:n]
    """
    cv = pipe.named_steps['cv']
    nb = pipe.named_steps['nb']
    feature_names = cv.get_feature_names_out()
    diff = nb.feature_log_prob_[0] - nb.feature_log_prob_[1]
    top_idx = np.argsort(diff)[::-1][:n]
    return [feature_names[i] for i in top_idx]


# 12. Bag of Words vs TF-IDF karşılaştırması
def compare_bow_vs_tfidf(X_train, X_test, y_train, y_test):
    """
    İki pipeline eğit ve test F1'lerini karşılaştır:
    - bow_f1:   CountVectorizer(stop_words='english', max_features=10000) + MultinomialNB
    - tfidf_f1: TfidfVectorizer(stop_words='english', max_features=10000) + MultinomialNB

    Amaç: bu problemde Bag of Words'ün (kelime sayımı) ne kadar güçlü bir
    baseline olduğunu TF-IDF'e karşı görmek.

    Returns:
        dict: {
            'bow_f1': float (CountVectorizer F1),
            'tfidf_f1': float (TfidfVectorizer F1)
        }

    İpucu:
    - from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    - İki ayrı Pipeline kur, fit et, pipe.predict(X_test) → f1_score.
    """
    results = {}
    pipelines = {
        'bow_f1': Pipeline([
            ('cv', CountVectorizer(stop_words='english', max_features=10000)),
            ('nb', MultinomialNB()),
        ]),
        'tfidf_f1': Pipeline([
            ('cv', TfidfVectorizer(stop_words='english', max_features=10000)),
            ('nb', MultinomialNB()),
        ]),
    }
    for key, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[key] = f1_score(y_test, y_pred, zero_division=0)
    return results


# 13. Toplu tahmin
def predict_batch(pipe, articles):
    """
    Birden çok haber için tahmin yap.

    Args:
        pipe: eğitilmiş pipeline
        articles: list of str (haber metinleri)

    Returns:
        list: her haber için predict_article çıktısı (list of dict)

    İpucu: [predict_article(pipe, a) for a in articles]
    """
    return [predict_article(pipe, a) for a in articles]



# 14. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Uçtan uca akış:
    1. fetch_news_data → load_news_data
    2. explore_data (fake_rate al)
    3. encode_labels
    4. split_data (X = df['text'], y = df['target'])
    5. build_bow_pipeline → train_model
    6. evaluate_model (test F1)
    7. top_fake_words (en fake kelime)
    8. predict_article ile iki örnek haber test et (biri fake, biri real)

    Örnek metinler (önerilen):
    - Fake: "SHOCKING! Secret elite cabal exposed, mainstream media won't tell
             you the truth they are hiding. Share before they delete this!"
    - Real: "The central bank announced on Tuesday that interest rates would
             remain unchanged, citing stable inflation and steady employment."

    Returns:
        dict: {
            'fake_rate': float,
            'test_f1': float,
            'top_fake_word': str (top_fake_words listesinin ilki),
            'sample_fake_pred': dict (predict_article çıktısı),
            'sample_real_pred': dict (predict_article çıktısı)
        }
    """
    # 1-2. Veri çek + yükle
    path = fetch_news_data()
    df = load_news_data(path)

    # 3. Keşif
    info = explore_data(df)

    # 4. Encode
    df = encode_labels(df)

    # 5. Split (X = haber metni, y = target)
    X_train, X_test, y_train, y_test = split_data(df['text'], df['target'])

    # 6-7. Pipeline + eğit
    pipe = build_bow_pipeline()
    pipe = train_model(pipe, X_train, y_train)

    # 8. Değerlendir
    metrics = evaluate_model(pipe, X_test, y_test)

    # 9-10. Örnek tahminler + en fake kelime
    fake_words = top_fake_words(pipe, n=10)
    sample_fake = predict_article(
        pipe,
        "SHOCKING! Secret elite cabal exposed, mainstream media won't tell you "
        "the truth they are hiding. Share before they delete this!",
    )
    sample_real = predict_article(
        pipe,
        "The central bank announced on Tuesday that interest rates would remain "
        "unchanged, citing stable inflation and steady employment figures.",
    )

    return {
        'fake_rate': info['fake_rate'],
        'test_f1': metrics['f1'],
        'top_fake_word': fake_words[0],
        'sample_fake_pred': sample_fake,
        'sample_real_pred': sample_real,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Fake rate        : {result['fake_rate']:.2%}")
    print(f"  Test F1          : {result['test_f1']:.4f}")
    print(f"  En fake kelime   : {result['top_fake_word']}")
    print(f"  Fake örnek tahmin: {result['sample_fake_pred']['label']} "
          f"(p={result['sample_fake_pred']['fake_probability']:.3f})")
    print(f"  Real örnek tahmin: {result['sample_real_pred']['label']} "
          f"(p={result['sample_real_pred']['fake_probability']:.3f})")
