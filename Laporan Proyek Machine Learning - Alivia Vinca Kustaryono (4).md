# Laporan Proyek Machine Learning - Alivia Vinca Kustaryono

## Project Overview

Industri anime telah berkembang pesat dalam dekade terakhir, dengan jutaan penggemar di seluruh dunia. Penonton anime dihadapkan dengan ribuan judul anime yang tersedia setiap tahun. Oleh karena itu, sistem rekomendasi anime menjadi kebutuhan penting untuk membantu penonton menemukan tayangan yang sesuai dengan preferensi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis machine learning menggunakan dataset *Anime Recommendations Database* dari Kaggle. Sistem ini bertujuan memberikan rekomendasi anime personal berdasarkan preferensi pengguna sebelumnya.

Rekomendasi yang akurat akan meningkatkan pengalaman pengguna, memperpanjang waktu menonton, dan meningkatkan loyalitas pengguna terhadap platform. Studi oleh \[Gomez-Uribe & Hunt, 2016] menunjukkan bahwa sistem rekomendasi menyumbang lebih dari 80% jam tayang Netflix, membuktikan efektivitas model personalisasi dalam meningkatkan kepuasan pengguna.

> Referensi:
> \[1]Schafer, J. B., Frankowski, D., Herlocker, J., & Sen, S. (2007). Collaborative Filtering Recommender Systems. In The Adaptive Web (pp. 291–324). Springer.
> \[2] Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. *ACM Transactions on Management Information Systems (TMIS)*, 6(4), 1–19.

## Business Understanding

### Problem Statements

1. Bagaimana cara memberikan rekomendasi anime yang relevan berdasarkan data pengguna yang dipersonalisasi dengan teknik content-based filtering?
2. Dengan data rating yang dimiliki, bagaimana cara untuk merekomendasikan anime yang akan disukai oleh penonton dan belum pernah ditonton oleh penonton tersebut sebelumnya?

### Goals

1. Membangun sistem rekomendasi anime yang mampu memberikan saran personal berbasis preferensi pengguna.
2. Menerapkan dan mengimplementasikan dua pendekatan sistem rekomendasi yaitu conten-based filtering dan collaborative filtering.

#### Solution Statement

Proyek ini menggunakan dua pendekatan solusi utama:
1. Content-Based Filtering
2. Collaborative Filtering

* **Pendekatan: Content-based Filtering**
Menggunakan metadata anime (seperti genre, tipe, dan rating) untuk membangun profil preferensi pengguna berdasarkan anime yang telah mereka beri rating tinggi. Model kemudian menghitung kesamaan antara profil pengguna dan fitur anime yang belum ditonton, menggunakan cosine similarity atau model pembobotan seperti TF-IDF. Rekomendasi diberikan berdasarkan kemiripan konten, bukan perilaku pengguna lain.

* **Pendekatan: Model-based Collaborative Filtering**
  Menggunakan neural collaborative filtering (NCF) dengan embedding layer untuk merepresentasikan pengguna dan anime dalam bentuk vektor laten, dan memprediksi rating berdasarkan arsitektur deep learning.

## Data Understanding

### Sumber Dataset
Dataset yang digunakan: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

### Deskripsi Dataset

Dataset ini berisi informasi tentang data preferensi pengguna dari 73.516 pengguna pada 12.294 anime. Setiap pengguna dapat menambahkan anime ke daftar lengkap mereka dan memberinya peringkat. Kumpulan data ini merupakan kompilasi dari peringkat tersebut.

Dataset terdiri dari dua file utama:

* `anime.csv` - berisi metadata anime.
* `rating.csv` - berisi rating yang diberikan pengguna terhadap anime.

#### Fitur pada `anime.csv`:

* `anime_id`: ID unik myanimelist.net yang mengidentifikasi anime
* `name`: Nama anime
* `genre`: Genre anime (dipisahkan dengan koma)
* `type`: Tipe anime (TV, Movie, dll)
* `episodes`: Episode dalam anime (1 jika movie)
* `rating`: Rata-rata rating untuk anime ini (dari -1 (tidak menonton) hingga 10)
* `members`: Jumlah anggota komunitas yang ada di "grup" anime ini.

#### Fitur pada `rating.csv`:

* `user_id`: ID pengguna
* `anime_id`: ID anime
* `rating`: Skor rating dari -1 (tidak menonton) hingga 10

### Exploratory Data Analysis (EDA)

**Cek Informasi Dataset**
(gambar anime.info())
dataset anime.csv memiliki 12.294 baris dimulai dari indeks 0 hingga 12.293. Dimana setiap baris merepresentasikan satu entri anime.
Dataset ini memiliki 7 kolom yaitu `anime_id`,`name`,`genre`,`type`,`episodes`,`rating`,`members`
Jumlah masing-masing tipe data:
* float64 → 1 kolom (`rating`)
* int64 → 2 kolom (`anime_id`, `members`)
* object → 4 kolom (`name`, `genre`, `type`, `episodes`)

(gambar rating.info())
dataset rating.csv memiliki 7.813.737 baris dimulai dari indeks 0 hingga 7.813.736. Dimana setiap baris merepresentasikan satu entri rating.
Dataset ini memiliki 3 kolom yaitu `user_id`,`anime_id`,`rating`
Jumlah masing-masing tipe data:
* int64 → 3 kolom (`user_id`,`anime_id`,`rating`)

**Statistik Deskriptif**
(gambar anime.describe())

Kolom: `anime`

| Statistik | Nilai     | Penjelasan                                                                 |
|-----------|-----------|-----------------------------------------------------------------------------|
| Count     | 12,294    | Jumlah entri/record anime yang tercatat                                    |
| Mean      | 14,058.22 | Rata-rata nilai ID anime                                                   |
| Std       | 11,455.29 | Sebaran ID anime (deviasi standar)                                         |
| Min       | 1         | ID anime terkecil                                                          |
| 25%       | 3,484.25  | 25% anime memiliki ID ≤ 3484           |
| Median    | 10,260.50 | Setengah anime memiliki ID ≤ 10260                                                      |
| 75%       | 24,794.50 | 75% anime memiliki ID ≤ 24794            |
| Max       | 34,527    | ID anime tertinggi                                                          |

Kolom: `rating`

| Statistik | Nilai  | Penjelasan                                                                 |
|-----------|--------|-----------------------------------------------------------------------------|
| Count     | 12,064 | Hanya ada 12.064 rating (230 missing/null)                                      |
| Mean      | 6.47   | Rata-rata rating adalah 6.47 artinya cukup positif                                           |
| Std       | 1.03   | Standar deviasi sekitar 1, rating cukup konsisten                                      |
| Min       | 1.67   | Rating terendah                                                             |
| 25%       | 5.88   | 25% anime memiliki rating ≤ 5.88               |
| Median    | 6.57   | Rating tengah adalah 6.57                                                         |
| 75%       | 7.18   | 75% anime memiliki rating ≤ 7.18                |
| Max       | 10.00  | Rating tertinggi                                                            |

Kolom: `members` 

| Statistik | Nilai       | Penjelasan                                                                 |
|-----------|-------------|-----------------------------------------------------------------------------|
| Count     | 12,294      | Tidak ada nilai kosong                    |
| Mean      | 18,071.34   | Rata-rata anime ditonton oleh 18 ribu orang                                         |
| Std       | 54,820.68   | Variasi yang sangat besar, artinya ada anime yang sangat populer|
| Min       | 5           | Anime paling tidak populer hanya ditonton oleh 5 orang                        |
| 25%       | 225         | 25% anime ditonton oleh ≤ 225 orang                                 |
| Median    | 1,550       | Setengah anime ditonton oleh ≤ 1.550 orang                                                |
| 75%       | 9,437       | 75% anime ditonton oleh ≤ 9.437 orang nilai ini                    |
| Max       | 1,013,917   | Anime terpopuler dengan penonton terbanyak                                  |


(gambar rating.describe())

Kolom: `user_id`

| Statistik | Nilai     | Penjelasan                                                             |
|-----------|-----------|-------------------------------------------------------------------------|
| Count     | 7,813,737 | Jumlah interaksi pengguna yang tercatat                                 |
| Mean      | 36,727.96 | Rata-rata nilai ID pengguna                                             |
| Std       | 20,997.95 | Sebaran nilai ID pengguna (deviasi standar)                             |
| Min       | 1         | ID pengguna terkecil                                                    |
| 25%       | 18,974.00 | 25% pengguna memiliki ID ≤ 18.974                                       |
| Median    | 36,791.00 | Setengah pengguna memiliki ID ≤ 36.791                                  |
| 75%       | 54,757.00 | 75% pengguna memiliki ID ≤ 54.757                                       |
| Max       | 73,516    | ID pengguna terbesar                                                    |

Kolom: `anime_id`

| Statistik | Nilai     | Penjelasan                                                             |
|-----------|-----------|-------------------------------------------------------------------------|
| Count     | 7,813,737 | Jumlah interaksi yang mencatat ID anime                                 |
| Mean      | 8,909.07  | Rata-rata nilai ID anime                                                |
| Std       | 8,883.95  | Deviasi standar besar, menunjukkan variasi ID yang luas                |
| Min       | 1         | ID anime terkecil                                                      |
| 25%       | 1,240.00  | 25% interaksi melibatkan anime dengan ID ≤ 1.240                        |
| Median    | 6,213.00  | Setengah interaksi melibatkan anime dengan ID ≤ 6.213                   |
| 75%       | 14,093.00 | 75% interaksi melibatkan anime dengan ID ≤ 14.093                       |
| Max       | 34,519    | ID anime terbesar                                                      |

Kolom: `rating`

| Statistik | Nilai     | Penjelasan                                                             |
|-----------|-----------|-------------------------------------------------------------------------|
| Count     | 7,813,737 | Jumlah rating yang diberikan pengguna                                   |
| Mean      | 6.14      | Rata-rata rating secara keseluruhan                                     |
| Std       | 3.73      | Deviasi besar karena ada rating -1 (belum memberikan rating)            |
| Min       | -1.00     | Rating minimum; -1 berarti user belum menilai anime                     |
| 25%       | 6.00      | 25% pengguna memberi rating ≤ 6.00                                      |
| Median    | 7.00      | Setengah rating berada di bawah atau sama dengan 7.00                   |
| 75%       | 9.00      | 75% rating berada di bawah atau sama dengan 9.00                        |
| Max       | 10.00     | Rating tertinggi yang bisa diberikan pengguna                           |



**Data Kosong**
(gambar anime.isnull().sum())
Terdapat 62 data kosong pada kolom `genre`, 25 data kosong pada kolom `type`, dan 230 data kosong pada kolom `rating`.

(gambar rating.isnull().sum())
Tidak terdapat data kosong pada rating.

**Data Duplikat**
(gambar anime.duplicated().sum() dan rating.duplicated().sum())
Tidak terdapat duplikat pada data anime dan terdapat satu duplikat pada data rating.

**Visualisasi Genre Anime**
(gambar Top 10 Genre Anime)
Dengan rincian sebagai berikut: Genre Komedi sebanyak 3193, Genre Action sebanyak 2845, Genre Sci-Fi sebanyak 1986, Genre Fantasy sebanyak 1815, Genre Shounen sebanyak 1663, Genre Adventure sebanyak 1457, Genre Comedy sebanyak 1452, Genre Romance sebanyak 1371, Genre Kids sebanyak 1213, dan Genre School sebanyak 1170.

**Visualisasi Tipe Anime**
(gambar Jenis-Jenis Anime berdasarkan Tipe)
Dengan rincian sebagai berikutt: TV sebanyak 3787, OVA sebanyak 3311, Movie sebanyak 2348, Special sebanyak 1676, ONA sebanyak 659, dan Music sebanyak 488.


## Data Preparation

Langkah-langkah preprocessing sebelum membangun sistem rekomendasi:

1. **Sampling rating sebanyak 50.000 rating** karena pada sebelumnya terdapat 7.813.737 baris jadi kita bisa mengambil sebanyak 50.000 sample dari rating tersebut.
2. **Membersihkan kolom kosong** pembersihan kolom kosong dilakukan dengan mengisi value pada kolom `genre` dengan 'Unknown', mengisi value pada kolom `type` dengan 'Unknown', dan mengisi value pada kolom `rating` dengan '0.00'. Hal ini dilakukan agar data variasi anime tetap dan tidak berkurang dan kolom kosong terisi.  
3. **Membersihkan nama anime** membersihkan nama anime dari karakter non-alfabetnumerik karena model akan lebih baik jika hanya bekerja dengan kata-kata yang bersih dan bermakna.
4. **Standarisasi genre anime** genre anime di standarisasi dan hanya di ambil satu genre pertama dari anime tersebut agar klasifikasi lebih akurat.

### Data Preparation & Preprocessing - Content Based Filtering
1. Ekstraksi Fitur dengan TF-IDF Untuk merepresentasikan setiap anime berdasarkan genre-nya. 
2. Selanjutnya transformasi genre ke bentuk matriks TF-IDF.
3. Selanjutnya ubah vektor TF-IDF ke dalam bentuk matriks todense untuk kebutuhan visual.
4. Selanjutnya buatlah dataframe untuk melihat matriks TF-IDF.

### Data Preparation & Preprocessing - Collaborative Filtering
1. Panggil kembali data rating yang sudah di sampling di awal `rating_sample`.
2. Selanjutnya filter rating dengan menghapus rating -1 karena menunjukkan anime yang belum di tonton.
3. Encode `user_id` dan `anime_id` menjadi index numerik.
4. Split data menjadi train dan validation set.

## Modeling
Pada proyek ini, dikembangkan sistem rekomendasi anime dengan dua pendekatan berbeda, yaitu content-based filtering dan collaborative filtering. Kedua metode ini bertujuan untuk menghasilkan rekomendasi anime yang dipersonalisasi berdasarkan preferensi pengguna.

### Content-Based Filtering
Content-Based Filtering adalah salah satu metode dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan kemiripan antar item berdasarkan fitur-fitur (atribut) item tersebut. Rekomendasi yang diberikan kepada pengguna adalah item yang memiliki kemiripan karakteristik dengan item yang pernah disukai atau dikonsumsi pengguna sebelumnya.

Contoh: Jika seorang pengguna menyukai film bergenre aksi, maka sistem akan merekomendasikan film lain yang juga bergenre aksi, memiliki pemeran serupa, atau disutradarai oleh sutradara yang sama.

#### Kelebihan dan Kekurangan Cosine Similiarity
**Kelebihan:**
- Tidak dipengaruhi oleh panjang vektor -- Fokus pada arah, bukan besar nilai, cocok untuk data teks.
- Efektif untuk high-dimensional sparse data -- Sangat umum dalam representasi data teks seperti TF-IDF.
- Mudah diimplementasikan dan dihitung secara efisien.

**Kekurangan:**
- Tidak mempertimbangkan magnitude -- Bisa jadi dua vektor sangat mirip arah tetapi memiliki bobot penting yang berbeda.
- Tidak memahami konteks atau semantik -- Jika digunakan pada data teks, dua kata yang berbeda tapi bermakna serupa bisa tetap dianggap tidak mirip.
- Kurang optimal untuk data non-linear atau kompleks -- Jika hubungan antar fitur tidak linier, cosine similarity bisa gagal menangkap kemiripan yang lebih kompleks.

#### Langkah-langkah Pembuatan Model Content-Based Filtering dengan Cosine Similarity
1. Persiapkan Data -- Siapkan data dengan fitur deskriptif dari item seperti genre anime.
2. Gabungkan fitur-fitur penting menjadi satu string -- gunakan metode TF-IDF untuk mengubah teks menjadi representasi vektor numerik
3. Gunakan Cosine Similarity untuk menghitung kemiripan antar item berdasarkan vektor fitur -- cosine similarity akan menghasilkan similarity matrix antar item.
4. Buat fungsi rekomendasi dengan input anime yang di suka dan output daftar anime yang paling mirip berdasarkan skor cosine similarity tertinggi.
5. Evaluasi sistem dengan test sistem, precision/recall, atau evaluasi manual.

### Collaborative Filtering
Collaborative Filtering adalah metode sistem rekomendasi yang memberikan rekomendasi kepada pengguna berdasarkan kesamaan preferensi atau perilaku pengguna lain. Metode ini tidak membutuhkan informasi fitur dari item atau pengguna, tetapi hanya mengandalkan interaksi seperti rating, klik, atau pembelian.

Contoh: Jika pengguna A dan B sama-sama menyukai produk 1 dan 2, dan A menyukai produk 3, maka sistem akan merekomendasikan produk 3 kepada B.

Terdapat dua jenis utama:
* User-Based: Mencari pengguna yang mirip dengan pengguna target.
* Item-Based: Mencari item yang mirip dengan item yang pernah disukai pengguna.

#### Kelebihan dan Kekurangan Collaborative Filtering
**Kelebihan:**
- Tidak perlu fitur konten -- Cocok saat tidak ada informasi detail tentang item.
- Personalized yaitu berdasarkan preferensi nyata dari pengguna.
- Menangkap pola kompleks yaitu dapat menemukan hubungan tak terduga antar pengguna/item.

**Kekurangan:**
- Cold Start Problem -- Sulit merekomendasikan untuk pengguna baru atau item baru karena belum ada interaksi.
- Data Sparsity -- Jika interaksi (rating/pembelian) jarang, model kesulitan menemukan pola.
- Scalability -- Komputasi bisa mahal untuk dataset besar tanpa optimasi (terutama user-based).

#### Langkah-langkah Pembuatan Collaborative Filtering Model dengan RecommenderNet (TensorFlow/Keras)
1. Persiapkan dataset yang berisi `user_id`,`anime_id`,`rating` lalu encode `user_id` dan `anime_id` menjadi angka untuk selanjutnya dibuat train-test split.
2. Buat arsitektur ReccomenderNet.
3. Lalu siapkan input data.
4. Setelah itu kompilasi dan training model.
5. Terakhir prediksi dan rekomendasi model.






## Evaluation

### Evaluasi Content Based Filtering
**Metrik Evaluasi Sistem Rekomendasi: Recommender System precision**
Precision mengukur seberapa relevan item yang direkomendasikan oleh sistem kepada pengguna.
Definisi:
$$
\text{Precision} = \frac{\text{Jumlah item relevan yang direkomendasikan}}{\text{Jumlah total item yang direkomendasikan}}
$$
Ini mencerminkan seberapa baik sistem dalam merekomendasikan item paling atas, yang paling mungkin dilihat pengguna.

### Evaluasi Collaborative Filtering
**Metrik Evaluasi Collaborative Filtering: MSE & RMSE**
Pada sistem rekomendasi ini, Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE) adalah metrik umum yang digunakan untuk mengevaluasi akurasi prediksi model karena keduanya mengukur rata-rata kuadrat atau akar kuadrat dari kesalahan (selisih antara nilai prediksi dan nilai sebenarnya).

**MSE (Mean Squared Error)**
MSE mengukur rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual.
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
* y i​ = rating aktual dari pengguna ke item
* 𝑦^𝑖y^​i​= rating yang diprediksi oleh model
* n = jumlah total prediksi

**RMSE (Root Mean Squared Error)**
RMSE adalah akar kuadrat dari MSE, yang artinya:
$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$
RMSE mengembalikan hasil dalam satuan yang sama dengan rating (misalnya 1–5), sehingga lebih mudah diinterpretasikan.

#### Hasil Evaluasi dengan MSE & RMSE

#### Visualisasi Ecpoch Tarining & Validation

## Recommendation Output

Model digunakan untuk memprediksi rating semua anime yang belum ditonton oleh user acak. Kemudian, 10 anime dengan prediksi rating tertinggi direkomendasikan.

Contoh output:

```
[Anime Favorit User]
- Attack on Titan | Genre: Action, Drama
- Naruto | Genre: Action, Shounen
...

[Rekomendasi Top 10 Anime]
- One Punch Man | Genre: Action, Comedy
- Death Note | Genre: Mystery, Supernatural
...
```

## Conclusion

Proyek ini berhasil membangun sistem rekomendasi anime berbasis neural collaborative filtering. Model menunjukkan performa yang stabil dan mampu menghasilkan rekomendasi yang relevan. Ke depan, sistem ini dapat ditingkatkan dengan:

* Menambahkan metadata (genre, tipe, studio) sebagai fitur tambahan.
* Menggunakan teknik regularisasi dan dropout untuk menghindari overfitting.
* Menguji dengan metrik top-N seperti Precision\@10, Recall\@10.

Sistem rekomendasi ini berpotensi diterapkan dalam platform streaming untuk meningkatkan keterlibatan dan kepuasan pengguna.
