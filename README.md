# PCA ve SVM ile Uçtan Uca Yüz Tanıma Projesi 🚀

Bu proje, **Labeled Faces in the Wild (LFW)** veri setini kullanarak, yüksek boyutlu görüntü verileri üzerinde makine öğrenmesi optimizasyon süreçlerini uygulamalı olarak ele almaktadır. Projede, boyut indirgeme yöntemi olan **PCA (Principal Component Analysis)** ve sınıflandırma algoritması olan **SVM (Support Vector Machines)** entegre bir pipeline mimarisinde bir araya getirilmiştir.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Python 3.x**
* **Scikit-Learn** (PCA, SVC, GridSearchCV, StandardScaler, Pipeline)
* **Numpy** & **Matplotlib** (Veri manipülasyonu ve görselleştirme)

---

## 📊 Proje Aşamaları ve Metodoloji

### 1. Görüntü Yeniden Yapılandırma (Image Reconstruction)
Her biri 1850 pikselden oluşan düzleştirilmiş yüz görüntüleri, PCA ile 150 temel bileşene sıkıştırılmıştır. Bilgi kaybının minimum düzeyde olduğunu doğrulamak amacıyla, sıkıştırılan veriler geri açılarak orijinal görüntülerle yan yana kıyaslanmıştır.

### 2. Eigenfaces (Ana Bileşen Analizi)
Veri setindeki maksimum varyansı temsil eden ilk 5 ana bileşen (en önemli yüz hatları/ışık yönleri) ile varyansı en düşük olan son 5 ana bileşen (gürültü/karıncalanma) görselleştirilerek PCA'in çalışma mantığı incelenmiştir.

### 3. Optimum Bileşen Sayısının Belirlenmesi
Kümülatif açıklanan varyans oranı (Cumulative Explained Variance) grafiği çizilerek, verideki bilgi çeşitliliğinin en az %80 ve %95'ini korumak için gereken minimum bileşen sayıları matematiksel olarak doğrulanmıştır.

### 4. Pipeline Yapısı ve Model Optimizasyonu
Veri sızıntısını (Data Leakage) önlemek amacıyla tüm süreç `scikit-learn` Pipeline mimarisine taşınmıştır:
* **Ölçeklendirme (Scaling):** `StandardScaler` kullanılarak piksel değerleri normalize edilmiş ve mesafe hassasiyeti olan SVM modelinin kararlılığı artırılmıştır.
* **Sınıf Dengeleme (Class Imbalance):** Veri setindeki dengesiz dağılım (bazı ünlülerin çok fazla, bazılarının çok az fotoğrafı olması durumu), SVC modelindeki `class_weight='balanced'` parametresiyle optimize edilmiştir.
* **Hiperparametre Ayarlama (Hyperparameter Tuning):** `GridSearchCV` ile en iyi `C`, `gamma` ve `kernel` kombinasyonları 3-katlı (3-fold) çapraz doğrulama ile taranarak nihai modele ulaşılmıştır.

---

## 📈 Sonuçlar ve Kazanımlar
* **Hız ve Hesaplama Maliyeti:** 1850 öznitelikten 150 bileşene düşülmesine rağmen modelin eğitim süreleri ciddi oranda kısalmış, işlem maliyeti düşürülmüştür.
* **Gürültü Filtreleme:** PCA'in düşük varyanslı pikselleri (gürültüleri) ayıklaması sayesinde, modelin test verisi üzerindeki genel genelleme (generalization) yeteneği artırılmıştır.