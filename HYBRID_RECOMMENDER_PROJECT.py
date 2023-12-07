
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
import pandas as pd
pd.pandas.set_option('display.max_columns', None) #tüm kolonları göster
pd.pandas.set_option('display.width', 300)

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv('/Users/melihasecilozturk/Desktop/miuul/ödevler/HybridRecommender-221114-235254/datasets/movie.csv')
movie.head()
movie.shape

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv('/Users/melihasecilozturk/Desktop/miuul/ödevler/HybridRecommender-221114-235254/datasets/rating.csv')
rating.head()
rating.shape

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df = movie.merge(rating, how="left", on="movieId") #movienin olduğunu sola at
df.head()

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
comment_counts = pd.DataFrame(df["title"].value_counts()) #her bir film için kaç kişi oy kullandı


# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = comment_counts[comment_counts["title"] <= 1000].index #1000den az rate alan filmler
common_movies = df[~df["title"].isin(rare_movies)]  #veri setinden çıkar


# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('/Users/melihasecilozturk/Desktop/miuul/ödevler/HybridRecommender-221114-235254/datasets/movie.csv')
    rating = pd.read_csv('/Users/melihasecilozturk/Desktop/miuul/ödevler/HybridRecommender-221114-235254/datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())  #her bir film kaç yorum almış
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index #1000 den az rate alan filmler,indx yazınca direkt isimlerini bi liste olarak verdi,çünkü indexteler
    common_movies = df[~df["title"].isin(rare_movies)] #yaygın film isimleri için nadir olanları çıkar
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
user_movie_df.head()


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user = int(pd.Series(user_movie_df.index).sample(1).values)
# 132665

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()  # #notna = NAN olmayanlar  #58 film


#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]   #[138493 rows x 58 columns]

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.notnull().sum() #T ile transpose unu aldık. filmler satır kulllanıcılar kolon
#oldu. True false yapacak izlediyse filmi 1 izlemediyse sıfır atacak. sonra 1 leri toplayınca. her bir kullanıcı kaç film
#izlemiş onu gösterir. T yaptık çünkü sum yukardan aşağıya çalışıyor.
user_movie_count = user_movie_count.reset_index() #user id leri index olmaktan çıkardık

user_movie_count.columns = ["userId", "movie_count"] #1. değişkenin adı user id 2. değişkenin adı movie count oldu


# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]



#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#corr_df[corr_df["user_id_1"] == random_user]



# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
#userımız en başta çıktığı için onun dışındakileri göster dedik
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
#her bir film için ortalama weighted rate bul ve ata
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
#movie id yi sütuna aldık
recommendation_df = recommendation_df.reset_index()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False).head(5)

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]



#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 132665

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv('/Users/melihasecilozturk/Desktop/miuul/ödevler/HybridRecommender-221114-235254/datasets/movie.csv')
rating = pd.read_csv('/Users/melihasecilozturk/Desktop/miuul/ödevler/HybridRecommender-221114-235254/datasets/rating.csv')
movie.head()
rating.info()
rating["timestamp"].sort_values()

#4182421     1995-01-09 11:46:44
#18950979    1995-01-09 11:46:49
#18950936    1995-01-09 11:46:49
#18950930    1995-01-09 11:46:49
#12341178    1996-01-29 00:00:00
#                   ...
#7819902     2015-03-31 06:00:51
#2508834     2015-03-31 06:03:17
#12898546    2015-03-31 06:11:26
#12898527    2015-03-31 06:11:28
#12675921    2015-03-31 06:40:02

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0] #[0:1] 0. indextekini ver
#82 movie id

#rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"]
#Out[94]:
#19177813     82
#19177856    593
#19177814    110
#19177849    527
#19177843    457
#19177808     35
#19177832    314
#19177837    345
#19177848    515
#19177846    497
#19177847    509
#19177806     17
#19177824    265
#19177833    318
#19177858    595
#19177817    150
#19177854    590
#Name: movieId, dtype: int64
#rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1]
#Out[95]:
#19177813    82
#Name: movieId, dtype: int64
#rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
#Out[96]: 82









# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
#movie csv yi kullandık
movie[movie["movieId"] == movie_id]["title"].values[0] #"Antonia's Line (Antonia) (1995)" title a çevir ki user movie df te kullan
#movie[movie["movieId"] == movie_id]["title"]
#Out[99]:
#81    Antonia's Line (Antonia) (1995)
#Name: title, dtype: object
#movie[movie["movieId"] == movie_id]["title"].values[0]
#Out[100]: "Antonia's Line (Antonia) (1995)"

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
# movie_df= user_movie_df["Antonia's Line (Antonia) (1995)"] üsttekinin aynı


user_movie_df.head()
movie_df.head()
# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)
#user_movie_df= kullanıcıların tüm filmlere verdikleri veya vermedikleri rateler
#movie_df= kullacıların Antonia's Line (Antonia) (1995) a verdiği veya vermediği rate

#title
#Antonia's Line (Antonia) (1995)                                         1.000000   kendiyle korelasyonu 1
#Accepted (2006)                                                         0.827954
#Nausicaä of the Valley of the Wind (Kaze no tani no Naushika) (1984)    0.641634
#Inside Job (2010)                                                       0.617387


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
oneri = user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)
oneri[1:6].index #kendisi hariç dedik o yüzden 1 den başladı .6. daki dahil deği. 1. 2. 3. 4. 5. indexler
#önerilecek filmler...
#index(['Accepted (2006)', 'Nausicaä of the Valley of the Wind (Kaze no tani no Naushika) (1984)', 'Inside Job (2010)', 'Futurama: Bender's Big Score (2007)', 'Paris, I Love You (Paris, je t'aime) (2006)'], dtype='object', name='title')




