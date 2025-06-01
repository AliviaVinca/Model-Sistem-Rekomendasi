# Laporan Proyek Machine Learning - Alivia Vinca Kustaryono

## Project Overview

Industri anime telah berkembang pesat dalam dekade terakhir, dengan jutaan penggemar di seluruh dunia. Penonton anime dihadapkan dengan ribuan judul anime yang tersedia setiap tahun. Oleh karena itu, sistem rekomendasi anime menjadi kebutuhan penting untuk membantu penonton menemukan tayangan yang sesuai dengan preferensi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis machine learning menggunakan dataset *Anime Recommendations Database* dari Kaggle. Sistem ini bertujuan memberikan rekomendasi anime personal berdasarkan preferensi pengguna sebelumnya.

Rekomendasi yang akurat akan meningkatkan pengalaman pengguna, memperpanjang waktu menonton, dan meningkatkan loyalitas pengguna terhadap platform. Studi oleh \[Gomez-Uribe & Hunt, 2016] menunjukkan bahwa sistem rekomendasi menyumbang lebih dari 80% jam tayang Netflix, membuktikan efektivitas model personalisasi dalam meningkatkan kepuasan pengguna.

> Referensi:

> \[1]Schafer, J. B., Frankowski, D., Herlocker, J., & Sen, S. (2007). Collaborative Filtering Recommender Systems. In The Adaptive Web (pp. 291–324). Springer.

> \[2] Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. *ACM Transactions on Management Information Systems (TMIS)*, 6(4), 1–19.

## Business Understanding

### Problem Statements

1. Bagaimana cara memberikan rekomendasi anime yang relevan berdasarkan preferensi pengguna menggunakan teknik content-based filtering?
2. Dengan data rating yang tersedia, bagaimana sistem dapat merekomendasikan anime yang belum pernah ditonton, namun kemungkinan besar disukai oleh pengguna?

### Goals

1. Membangun sistem rekomendasi anime yang mampu memberikan saran personal berdasarkan preferensi pengguna.
2. Mengimplementasikan dua pendekatan sistem rekomendasi, yaitu content-based filtering dan collaborative filtering.

#### Solution Statement

Proyek ini menggunakan dua pendekatan solusi utama:
1. Content-Based Filtering
2. Collaborative Filtering

* **Pendekatan: Content-based Filtering**

Pendekatan ini menggunakan metadata dari anime (seperti genre, tipe, dan rating) untuk membentuk profil preferensi pengguna. Profil ini dibangun berdasarkan anime yang sebelumnya diberi rating tinggi oleh pengguna. Model kemudian menghitung kesamaan antara profil pengguna dan fitur dari anime yang belum ditonton menggunakan teknik seperti cosine similarity atau pembobotan TF-IDF.
Rekomendasi diberikan berdasarkan kemiripan konten, tanpa bergantung pada data pengguna lain.

* **Pendekatan: Model-based Collaborative Filtering**
  
Menggunakan pendekatan model-based collaborative filtering dengan Neural Collaborative Filtering (NCF). Pendekatan ini memanfaatkan embedding layer untuk merepresentasikan pengguna dan anime dalam bentuk vektor laten, dan memprediksi rating menggunakan arsitektur deep learning.
Model ini belajar dari pola interaksi antar pengguna dan item, dan mampu menangkap hubungan kompleks antar entitas.
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

#### **Cek Informasi Dataset**

![Screenshot 2025-06-01 084157](https://github.com/user-attachments/assets/66a9a898-f45e-4498-89d3-41a3ac9f78d9)

(gambar anime.info())

dataset anime.csv memiliki 12.294 baris dimulai dari indeks 0 hingga 12.293. Dimana setiap baris merepresentasikan satu entri anime.
Dataset ini memiliki 7 kolom yaitu `anime_id`,`name`,`genre`,`type`,`episodes`,`rating`,`members`
Jumlah masing-masing tipe data:
* float64 → 1 kolom (`rating`)
* int64 → 2 kolom (`anime_id`, `members`)
* object → 4 kolom (`name`, `genre`, `type`, `episodes`)

![Screenshot 2025-06-01 084220](https://github.com/user-attachments/assets/7411b9c0-0ff4-4c3b-ae54-86d39d14a65b)

(gambar rating.info())

dataset rating.csv memiliki 7.813.737 baris dimulai dari indeks 0 hingga 7.813.736. Dimana setiap baris merepresentasikan satu entri rating.
Dataset ini memiliki 3 kolom yaitu `user_id`,`anime_id`,`rating`
Jumlah masing-masing tipe data:
* int64 → 3 kolom (`user_id`,`anime_id`,`rating`)


#### **Statistik Deskriptif**

![Screenshot 2025-06-01 084240](https://github.com/user-attachments/assets/25642e0e-608b-4350-8b49-93eef9b8ad72)

(gambar anime.describe())

Kolom: `anime`

| Statistik | Nilai     | Penjelasan                                                                 |
|-----------|-----------|-----------------------------------------------------------------------------|
| Count     | 12,294    | Jumlah entri/record anime yang tercatat                                    |
| Mean      | 14,058.22 | Rata-rata nilai ID anime                                                   |
| Std       | 11,455.29 | Sebaran ID anime (deviasi standar)                                         |
| Min       | 1         | ID anime terkecil                                                          |
| 25%       | 3,484.25  | 25% anime memiliki ID ≤ 3484           |
| Median    | 10,260.50 | Setengah anime memiliki ID ≤ 10260                                         |
| 75%       | 24,794.50 | 75% anime memiliki ID ≤ 24794            |
| Max       | 34,527    | ID anime tertinggi                                                          |

Kolom: `rating`

| Statistik | Nilai  | Penjelasan                                                                 |
|-----------|--------|-----------------------------------------------------------------------------|
| Count     | 12,064 | Hanya ada 12.064 rating (230 missing/null)                                  |
| Mean      | 6.47   | Rata-rata rating adalah 6.47 artinya cukup positif                          |
| Std       | 1.03   | Standar deviasi sekitar 1, rating cukup konsisten                           |
| Min       | 1.67   | Rating terendah                                                             |
| 25%       | 5.88   | 25% anime memiliki rating ≤ 5.88               |
| Median    | 6.57   | Rating tengah adalah 6.57                                                   |
| 75%       | 7.18   | 75% anime memiliki rating ≤ 7.18                |
| Max       | 10.00  | Rating tertinggi                                                            |

Kolom: `members` 

| Statistik | Nilai       | Penjelasan                                                                 |
|-----------|-------------|-----------------------------------------------------------------------------|
| Count     | 12,294      | Tidak ada nilai kosong                    |
| Mean      | 18,071.34   | Rata-rata anime ditonton oleh 18 ribu orang                                |
| Std       | 54,820.68   | Variasi yang sangat besar, artinya ada anime yang sangat populer|
| Min       | 5           | Anime paling tidak populer hanya ditonton oleh 5 orang                     |
| 25%       | 225         | 25% anime ditonton oleh ≤ 225 orang                                 |
| Median    | 1,550       | Setengah anime ditonton oleh ≤ 1.550 orang                                  |
| 75%       | 9,437       | 75% anime ditonton oleh ≤ 9.437 orang nilai ini                    |
| Max       | 1,013,917   | Anime terpopuler dengan penonton terbanyak                                  |


![Screenshot 2025-06-01 084251](https://github.com/user-attachments/assets/f8ec01d7-2323-4967-b703-a8b75a18665d)

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



#### **Data Kosong**

![Screenshot 2025-06-01 084308](https://github.com/user-attachments/assets/6e7d5009-9a02-4a8d-97f2-91e46603d2b3)

(gambar anime.isnull().sum())

Terdapat 62 data kosong pada kolom `genre`, 25 data kosong pada kolom `type`, dan 230 data kosong pada kolom `rating`.

![Screenshot 2025-06-01 084316](https://github.com/user-attachments/assets/71210f41-ad86-4c82-ab97-c4dda0eb6a5b)

(gambar rating.isnull().sum())

Tidak terdapat data kosong pada rating.


#### **Data Duplikat**

![Screenshot 2025-06-01 084345](https://github.com/user-attachments/assets/abce728f-82cb-4ec2-872e-e260ceae4b56)

(gambar anime.duplicated().sum() dan rating.duplicated().sum())

Tidak terdapat duplikat pada data anime dan terdapat satu duplikat pada data rating.


#### **Visualisasi Genre Anime**

![top 10 genre anime](https://github.com/user-attachments/assets/3fa3acf5-7f3d-4ee4-95a7-fb1b4aaf21ec)

(gambar Top 10 Genre Anime)

Dengan rincian sebagai berikut: Genre Komedi sebanyak 3193, Genre Action sebanyak 2845, Genre Sci-Fi sebanyak 1986, Genre Fantasy sebanyak 1815, Genre Shounen sebanyak 1663, Genre Adventure sebanyak 1457, Genre Comedy sebanyak 1452, Genre Romance sebanyak 1371, Genre Kids sebanyak 1213, dan Genre School sebanyak 1170.

#### **Visualisasi Tipe Anime**

![tipe anime](https://github.com/user-attachments/assets/d93be0b5-fc7c-4c3d-89e5-5f37d28ebb41)

(gambar Jenis-Jenis Anime berdasarkan Tipe)

Dengan rincian sebagai berikutt: TV sebanyak 3787, OVA sebanyak 3311, Movie sebanyak 2348, Special sebanyak 1676, ONA sebanyak 659, dan Music sebanyak 488.


## Data Preparation

Langkah-langkah preprocessing sebelum membangun sistem rekomendasi:

1. **Sampling Data Rating**
- Langkah: Mengambil 500.000 baris secara acak dari dataset `rating.csv`.
- Alasan: Dataset `rating.csv` terdiri dari 7.813.737 entri, yang terlalu besar dan membutuhkan waktu pemrosesan lama. Oleh karena itu, dilakukan sampling agar pemodelan lebih efisien namun tetap representatif.
   
2. **Penanganan Nilai Kosong**
- Langkah:
  - Kolom `genre` diisi dengan `'Unknown'`
  - Kolom `type` diisi dengan `'Unknown'`
  - Kolom `rating` (dari `anime.csv`) diisi dengan `0.00`
- Alasan: Menghindari hilangnya entri anime yang valid hanya karena terdapat data kosong. Dengan mengisi nilai kosong, variasi data tetap terjaga tanpa kehilangan baris penting.

3. **Pembersihan Nama Anime**
- Langkah: Menghapus karakter non-alfanumerik dari kolom name.
- Alasan: Nama anime yang bersih mempermudah pemrosesan teks dalam model content-based filtering, terutama saat dilakukan pencarian judul mirip.

### Data Preparation & Preprocessing - Content Based Filtering
1. **Standarisasi Genre**
- Langkah: Mengubah tanda pemisah genre dari koma (,) menjadi spasi.
- Alasan: Genre yang rapi dan konsisten akan menghasilkan representasi fitur yang lebih baik saat diekstraksi menggunakan TF-IDF.

2. Ekstraksi Fitur dengan TF-IDF
- Langkah:
  - Menggunakan TF-IDF Vectorizer pada kolom genre.
  - Mengubah genre menjadi matriks fitur berbobot.
  - Mengubah matriks ke bentuk dense untuk keperluan analisis visual.
  - Membuat DataFrame dari hasil vektorisasi untuk interpretasi.
- Alasan: TF-IDF membantu merepresentasikan kemiripan antar anime berdasarkan genre, yang penting dalam sistem rekomendasi berbasis konten.

### Data Preparation & Preprocessing - Collaborative Filtering
1. Pemanggilan dan Pembersihan Rating
- Langkah:
  - Menggunakan rating_sample dari hasil sampling awal.
  - Menghapus rating bernilai -1 (menandakan belum menonton).
- Alasan: Rating -1 tidak dapat digunakan untuk melatih model karena tidak memberikan informasi preferensi pengguna.

2. Encoding ID
- Langkah: Mengubah user_id dan anime_id menjadi ID numerik kontinu menggunakan LabelEncoder.
- Alasan: Model Machine Learning (terutama model matriks embedding) hanya dapat bekerja dengan input numerik.

3. Split Dataset
- Langkah: Membagi dataset menjadi training set dan validation set.
- Alasan: Agar dapat melatih model dan menguji performanya secara objektif menggunakan data yang tidak dilihat saat pelatihan.

## Modeling
Pada proyek ini dikembangkan sistem rekomendasi anime yang bertujuan memberikan rekomendasi yang sesuai dengan minat pengguna. Dua pendekatan utama yang digunakan adalah Content-Based Filtering dan Collaborative Filtering, dengan harapan mampu menangani berbagai kondisi seperti cold-start atau data sparsitiy secara komplementer.

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

#### Top 5 Recomendation (Content-Based Filtering)
Untuk pendekatan Content-Based Filtering, hasil rekomendasi didapatkan dengan cara mengamati kualitas Top-N rekomendasi berdasarkan anime yang pernah ditonton atau disukai pengguna.

Ketika pengguna sebelumnya menyukai anime "Saiki Kusuo no nan TV" dengan klasifikasi sebagai berikut :

![Screenshot 2025-06-01 233441](https://github.com/user-attachments/assets/ab2c90ac-385d-47be-a3c2-1a41f089e16f)

Maka sistem akan merekomendasikan 5 anime berikut :

![Screenshot 2025-06-01 233643](https://github.com/user-attachments/assets/cfa45d99-0789-4d5e-a36c-e1dbc7dae98f)

Rekomendasi tersebut relevan karena memiliki genre yang sama atau hampir mirip yaitu Comedy, School, Shounen, Supernatural.

Kita juga dapat menambahkan sistem Fuzzy dalam Content-Based Filtering ini agar bisa memasukan separuh judul animenya lalu langsung terpanggil judul anime terdekat dan rekomendasi animenya. 

Contohnya seperti di bawah ini. Saat kita baru memanggil "Sword Art On" langsung ditampilkan rekomendasi untuk anime "Sword Art Online".

![Screenshot 2025-06-01 234155](https://github.com/user-attachments/assets/dd7c540f-c9f9-49e5-981b-c23bfed81a2e)

![Screenshot 2025-06-01 234223](https://github.com/user-attachments/assets/db831039-d51e-429f-bceb-6f082f8c913a)

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

#### Top 10 Recomendation (Collaborative Filtering)
Untuk pendekatan Collaborative Filtering, sistem memberikan Top-N rekomendasi film untuk pengguna tertentu berdasarkan skor kecocokan. Misalnya user 35778 yang memberikan rating tinggi pada anime berikut :

![Screenshot 2025-06-02 002649](https://github.com/user-attachments/assets/3adec6d2-97bf-4a55-969d-75d99225154a)

Maka sistem akan merekomendasikan 10 anime berikut:
![Screenshot 2025-06-02 002753](https://github.com/user-attachments/assets/50d36e8e-0283-4966-bc76-f20e98c04f4a)

Rekomendasi tersebut relevan dengan preferensi pengguna, terutama karena banyaknya anime yang memiliki genre mirip dengan anime yang sebelumnya disukai oleh pengguna. Hal ini menunjukkan bahwa model collaborative filtering berhasil mempelajari pola ketertarikan pengguna berdasarkan perilaku pengguna lain yang mirip.

### Kesimpulan Modeling and Result
Hasil rekomendasi menunjukkan bahwa sistem mampu mengidentifikasi anime dengan genre atau preferensi yang relevan berdasarkan histori pengguna. Meskipun belum dilakukan evaluasi kuantitatif seperti precision/recall, validasi visual terhadap top-N rekomendasi menunjukkan kualitas yang memuaskan.

Kedua pendekatan saling melengkapi: content-based filtering efektif saat data pengguna masih terbatas, sementara collaborative filtering mampu menangkap pola preferensi pengguna secara lebih dalam. Kombinasi keduanya membuka peluang untuk membangun sistem rekomendasi hybrid di masa depan.


## Evaluation

### Evaluasi Content Based Filtering

**Metrik Evaluasi: Precision@K**

Precision mengukur seberapa banyak item yang direkomendasikan oleh sistem benar-benar relevan bagi pengguna. Metrik ini fokus pada kualitas rekomendasi, terutama pada Top-K rekomendasi yang ditampilkan.

Formula:

$\text{Precision@K} = \frac{\text{Jumlah item relevan yang direkomendasikan (dalam Top-K)}}{\text{Jumlah total item yang direkomendasikan(K)}}$

Precision@K sangat cocok digunakan ketika kita hanya ingin menampilkan sejumlah rekomendasi teratas yang paling relevan dan ingin menghindari item tidak relevan.

#### Hasil Evaluasi dengan Precision
**Kita akan mencoba Precision menggunakan `anime_recomendation`**

Mari kita ambil 1 user fiktif, yang menyukai (menonton dan memberi rating tinggi) beberapa anime lalu panggil precision nya.

![Screenshot 2025-06-02 001631](https://github.com/user-attachments/assets/7764374b-20bc-4d9b-8681-255fb14cf8da)

Model Content-Based Filtering dievaluasi menggunakan metrik Precision@5 terhadap anime Saiki Kusuo no nan TV sebagai input. Dari 5 rekomendasi yang dihasilkan, terdapat X anime yang sesuai dengan anime yang disukai pengguna (ground truth), sehingga:

$\text{Precision@5} = \frac{\text{X}}{\text{5}}=0.X$

**Kita akan mencoba Precision dengan manual**

Kita uji dengan contoh user fiktif yang menyukai anime tertentu. Sistem merekomendasikan 5 anime teratas berdasarkan kemiripan konten.

📌 Contoh Kasus:
- Input: "Saiki Kusuo no Ψ-nan TV"
- Output: 5 anime teratas berdasarkan cosine similarity.

![Screenshot 2025-06-02 000604](https://github.com/user-attachments/assets/51e19244-cb8e-436e-a2f9-a7f6d8336d1c)

Dari hasil rekomendasi:

2 dari 5 anime ternyata sesuai dengan preferensi pengguna berdasarkan histori.

$\text{Precision@5} = \frac{\text{2}}{\text{5}}=0.4$

Diperoleh nilai Precision@5 = 0.4

Lalu kita coba menguji dengan 2 pengguna uji coba (Multi User).

![Screenshot 2025-06-02 000637](https://github.com/user-attachments/assets/5040b2fc-3f65-44fd-b0fb-c574a40c10ff)

Berdasarkan hasil evaluasi Content-Based Filtering menggunakan metrik Precision@5 terhadap 2 pengguna uji coba, diperoleh nilai sebagai berikut:
- User 1: Precision@5 = 0.4
- User 2: Precision@5 = 0.4

Rata-rata Precision@5

$\frac{\text{0.4 + 0.4}}{\text{2}}=0.4$

Maka, rata-rata Precision@5 adalah 0.40. Nilai ini menunjukkan bahwa dari 5 anime yang direkomendasikan, rata-rata 2 dari 5 adalah anime yang memang relevan atau disukai pengguna berdasarkan histori.

### Evaluasi Collaborative Filtering
**Metrik Evaluasi Collaborative Filtering: MSE & RMSE**
Pada sistem rekomendasi ini, Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE) adalah metrik umum yang digunakan untuk mengevaluasi akurasi prediksi model karena keduanya mengukur rata-rata kuadrat atau akar kuadrat dari kesalahan (selisih antara nilai prediksi dan nilai sebenarnya).

**MSE (Mean Squared Error)**

MSE mengukur rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual.

$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $

* `yᵢ`= rating aktual dari pengguna ke item
* `ŷᵢ`​= rating yang diprediksi oleh model
* `n` = jumlah total prediksi


**RMSE (Root Mean Squared Error)**

RMSE adalah akar kuadrat dari MSE, yang artinya:

$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } = \sqrt{MSE}$

RMSE mengembalikan hasil dalam satuan yang sama dengan rating (misalnya 1–5), sehingga lebih mudah diinterpretasikan.

#### Hasil Evaluasi dengan MSE & RMSE
![Screenshot 2025-06-02 002035](https://github.com/user-attachments/assets/e3aec390-5a95-43aa-8323-f4c0e4f2f48a)

Hasil evaluasi Collaborative Filtering menunjukkan performa model pada data validasi dengan nilai loss (MSE) sebesar sekitar 36.3658 dan RMSE sekitar 6.0290. Model yang dievaluasi memiliki tingkat kesalahan prediksi yang tergolong sedang, karena nilai RMSE berada di antara 5 hingga 7. Ini berarti rata-rata kesalahan prediksi rating model berkisar ±6 poin pada skala rating yang digunakan. Meskipun model sudah belajar pola dari data pelatihan, masih terdapat ruang untuk perbaikan dalam meminimalkan kesalahan prediksi, terutama untuk meningkatkan kualitas rekomendasi pada data yang belum terlihat sebelumnya.

#### Visualisasi Ecpoch Tarining & Validation
![RMSE (2)](https://github.com/user-attachments/assets/8a4d913c-b060-4b3e-983a-54aee92b3ed0)

Kurva biru (Train RMSE) menunjukkan bahwa error pada data pelatihan menurun cukup stabil dari awal hingga akhir pelatihan, dengan RMSE yang perlahan turun dari sekitar 4.6 ke sekitar 2.0. Ini menunjukkan bahwa model berhasil belajar dan menyesuaikan diri terhadap data pelatihan secara bertahap.

Sementara itu, kurva oranye (Validation RMSE) justru mengalami tren yang meningkat seiring bertambahnya epoch. RMSE validasi naik perlahan dari sekitar 2.0 menjadi lebih dari 2.5 di akhir pelatihan. Pola ini menunjukkan bahwa model mulai mengalami overfitting, yaitu kondisi ketika model terlalu menyesuaikan diri dengan data pelatihan dan kehilangan kemampuan generalisasi terhadap data baru (validasi).

Dengan kata lain, meskipun performa pada data pelatihan semakin baik, performa pada data validasi justru menurun, menandakan bahwa model tidak bekerja optimal untuk merekomendasikan item kepada pengguna yang belum ada dalam data pelatihan.


## Output

### Content-Based Filtering
Model digunakan untuk memprediksi rekomendasi anime mirip dengan anime yang di panggil.

![Screenshot 2025-06-01 183701](https://github.com/user-attachments/assets/189aa5c3-7f7f-4aad-8d01-7db24ebd8a78)

### Collaborative Filtering

Model digunakan untuk memprediksi rating semua anime yang belum ditonton oleh user acak. Kemudian, 10 anime dengan prediksi rating tertinggi direkomendasikan.


```
Showing recommendations for user: 35778
===========================
Anime with high ratings from user
--------------------------------
Fullmetal Alchemist Brotherhood : Action Adventure Drama Fantasy Magic Military Shounen
Tonari no Totoro : Adventure Comedy Supernatural
Evangelion 10 You Are Not Alone : Action Mecha Sci-Fi
Naruto : Action Comedy Martial Arts Shounen Super Power
Houkago no Pleiades TV : Magic Space
--------------------------------
Top 10 anime recommendation
--------------------------------
Kimi no Na wa : Drama Romance School Supernatural
Gintama : Action Comedy Historical Parody Samurai Sci-Fi Shounen
SteinsGate : Sci-Fi Thriller
Gintama039 : Action Comedy Historical Parody Samurai Sci-Fi Shounen
Hunter x Hunter 2011 : Action Adventure Shounen Super Power
Gintama Movie Kanketsuhen  Yorozuya yo Eien Nare : Action Comedy Historical Parody Samurai Sci-Fi Shounen
Clannad After Story : Drama Fantasy Romance Slice of Life Supernatural
Code Geass Hangyaku no Lelouch R2 : Action Drama Mecha Military Sci-Fi Super Power
Shigatsu wa Kimi no Uso : Drama Music Romance School Shounen
Tengen Toppa Gurren Lagann : Action Adventure Comedy Mecha Sci-Fi
```

## Conclusion

Proyek ini berhasil membangun sistem rekomendasi anime berbasis Content-Based Filtering dan Collaborative Filtering. Kesimpulan yang dapat kita ambil adalah:
- Content-Based Filtering bekerja baik untuk user dengan preferensi genre tertentu.
- Collaborative Filtering efektif untuk data interaksi tinggi, tapi rawan overfitting.

Model mampu menghasilkan rekomendasi yang relevan. Ke depan, sistem ini dapat ditingkatkan dengan:

* Menambahkan metadata (genre, tipe, studio) sebagai fitur tambahan.
* Menggunakan teknik regularisasi dan dropout untuk menghindari overfitting.
* Tambahkan hybrid approach, lebih banyak user rating, regularisasi model.
* Menguji dengan metrik top-N seperti Precision\@10, Recall\@10.

Sistem rekomendasi ini berpotensi diterapkan dalam platform streaming untuk meningkatkan keterlibatan dan kepuasan pengguna.
