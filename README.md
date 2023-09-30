# MLAlgorithms
KNN, Perceptrons, Decision Tree, Naive Bayes (from scratch and with scikit-learn)

Celem projektu jest przeprowadzenie analizy dowolnie wybranego zbioru danych, przy 
pomocy samodzielnie napisanych programów algorytmów uczenia maszynowego typu: KNN, 
perceptronu. Następnie na podstawie przeprowadzonych testów, należy utworzyć macierz 
pomyłek oraz obliczyć dokładność predykcji poszczególnych algorytmów.

Baza danych wykorzystana w poniższej pracy to zbiór o nazwie Glass Identification. 
Zbiór zawiera 7 różnych typów szkła, określonych za pomocą 7 atrybutów, które zawierają 
pierwiastki zawarte w składzie szkła. Pierwiastki opisane są procentowym udziałem w związku. 
Docelowo wyniki badań tej bazy danych, mają pomóc w ustaleniu rodzaju szkła na podstawie 
jego składu. 

Po wczytaniu zbioru głównego z bazą szkła, przy pomocy funkcji z biblioteki pandas 
read_csv, został on „przeshuflowany” funkcją sample() [1]. Następnie zbiór został podzielony 
na podzbiory treningowy, testowy i walidacyjny w odpowiedniej propocji 60:20:20 [2]. Zbiór 
treningowy posłużył do uczenia algorytmów, testowy do jednorazowego testowania ich 
działania, natomiast walidacyjny posłużył do finalnego sprawdzenia działania algorytmów. 
Podzbiory zostały wyeksportowane do oddzielnych plików csv, tak aby można było je wczytać 
i aby nie ich wartości nie zmieniały się za każdym uruchomieniem programu.

Z racji dużej różnicy pomiędzy wartościami niektórych atrybutów, zostały one 
przeskalowane tak aby zawierały się w przedziale 0-2. Skalowanie dotyczyło dokładnie kolumn 
z atrybutami: Mg, Na, Si, Ca. [3]

Do porównania efektywności użytych w projekcie algorytmów, wykorzystano macierz 
pomyłek oraz dokładność procentową wyników. Macierz pomyłek to macierz 7x7 [4]
zawierająca predykcje każdej z instancji. Kolumny stanowią wartości prawdziwe „real” R1,
czyli target naszej bazy, natomiast wiersze określane są poprzez wartości przewidywane 
„predicted” P1, czyli wartości obliczone przez algorytm. Mając wynik predykcji oraz nasz 
target układamy współrzędne [target][predykcja], w celu odnalezienia danego miejsca w tabeli 
pomyłek oraz dodaniu wartości 1.

Dokładność algorytmu wyliczana jest za pomocą jednej linijki kodu na podstawie, 
której predykcji porównywane są do naszego targetu, gdy obie wartości są identyczne, to 
wtedy zwracana jest jedynka. Wszystkie jedynki dla wszystkich instancji są zsumowane i 
dzielone przez liczbę naszych targetów co daje nam procentową dokładność.


W projekcie zostały wykorzystane następujące algorytmy: KNN (k najbliższych
sąsiadów), Perceptron one vs rest, MultiLayerPerceptron, Decision Tree, Naive Bayesa.

Algorytm KNN

Analizuje obiekty umieszczone w sąsiedztwie badanej próbki. Algorytm bierze po kolei 
każdą z próbek, a następnie wylicza odległość euklidesową od pozostałych obiektów. 
Odległości te są sortowane w kolejności rosnącej, a następnie wyłaniane jest K sąsiadów o 
najmniejszej odległości. Estymowana próbka zostaje przyporządkowana do klasy, która 
występują najczęściej w zbiorze K sąsiadów.

Perceptron OVR

W celu przeprowadzenia analizy perceptronem OVR, istnieje konieczność stworzenia n 
perceptronów odpowiadających liczbie klas naszego zbioru. Każdy z perceptronów poddawany 
jest uczeniu, w którym dla perceptronu o n klasie, dla targetu w miejsce klasy n wstawiane są 
jedynki, w pozostałych przypadkach mamy doczynienia z zerami. Uczenie zachodzi przez 
określoną liczbę iteracji. Podczas niego obliczany jest wektor skalarny pomiędzy wagami, a 
atrybutami próbki. Następnie funkcja step zwraca wartość 1 w przypadku gdy wektor jest 
większy od 0. W każdej iteracji parametry algorytmu typu waga oraz bias są aktualizowane o 
wartość obliczoną ze wzoru: learning_rate * (target – oszacowana_klasa). Po zakończeniu 
iteracji uczenia, możemy przejść do predykcji, w której wykorzystujemy wagi oraz bias z 
poprzedniego etapu, do wyliczenia szacowanych wartości bazy.

MultiLayerPerceptron

Algorytm ten oparty jest o działanie powyżej opisanego perceptronu. Sieć ta składa się 
z wielu warstw pojedynczych neuronów, w taki sposób że wyjścia neuronów warstwy 
poprzedniej tworzą wektor podawany na wejście każdego z neuronów warstwy następnej.
Ostatnia warstwa sieci to warstwa wyjściowa, w której neuronach formowane są sygnały 
wyjściowe. Liczba neuronów w tej warstwie najczęściej jest równa liczbie klas.

Decision Tree

Algorytm polega na tworzeniu wewnętrznych węzłów odpowiadających 
przeprowadzeniu testów na wartościach atrybutów. Z węzła wewnętrznego wychodzi tyle 
gałęzi, ile jest możliwych wyników testu, każdy liść zawiera decyzję o klasyfikacji próbki. 
Algorytm generuje węzły wraz z rozdzieleniem danych treningowych do momentu, w którym 
do węzła należeć będą wyłącznie przykłady przydzielone do jednej klasy decyzyjnej.

Naive Bayes 

W przypadku klasyfikacji za pomocą algorytmu Naive Bayesa, stawiamy dość 
niecodzienne założenia początkowe m.in., że wszystkie cechy wejściowe są tak samo ważne i 
niezależne od siebie. Ucząc klafysikatora bayesowskiego, tworzymy model statystyczny.
Wyliczana zostaje wartość prawdopodobieństwa wystąpienia każdej z klas. 

Poniżej zamieszczone zostały tablice pomyłek wraz z dokładnością obliczeń dla 
poszczególnych algorytmów. Oprócz tego zbadano również wpływ parametrów na wyniki 
estymacji. Obliczenia przeprowadzone zostały na zbiorze walidacyjnym. Target naszego zbioru 
prezentuje się następująco: 
1 2 5 2 7 2 2 5 1 2 3 2 2 2 7 2 7 1 3 2 1 2 5 2 6 2 1 2 1 1 1 1 1 1 1 7 2 1 5 2 3 1 2 1.

Algorytm KNN

Pierwszym przebadanym algorytmem był algorytm KNN z ustawionym parametrem K 
= 4. Dokładność obliczeniowa w tym przypadku wyniosła 67%

![image](https://github.com/washaq00/MLAlgorithms/assets/109302076/102c79d0-f334-48be-9e10-75d9f032dcb5)

Tablica pomyłek dla algorytmu KNN k=4
Następnie sprawdzono wpływ parametru K na dokładność procentową wyniku:
[7]. Wykres zależności dokładności obliczeniowej algorytmu knn od parametru.
Z wykresu można wywnioskować, że zwiększenie liczby sąsiadów K, zmniejsza 
dokładność obliczeniową algorytmu w dużym stopniu. Natomiast, największą dokładność 
obliczeniową obserwujemy w przypadku k równego 4.
Perceptron OVR
Dla perceptronu OVR zastosowano parametry learning_rate = 0.5 oraz liczbę iteracji= 
1000. Jego dokładność obliczeniowa wyniosła 34%. Zmiana parametrów w przypadku tego 
algorytmu nie wnosi znaczących zmian. Może być to spowodowane specyficzną bazą danych.
[8]. Tablica pomyłek dla perceptronu. 
MultiLayerPerceptron
W warunkach początkowych tego perceptronu została ustawiona maksymalna liczba 
iteracji = 200. Dokładność algorytmu dla takiego parametru wyniosła 54 %. Następnie zbadano 
zależność dokładności od liczby iteracji w zakresie od 100 do 500 iteracji.
[9]. Tablica pomyłek dla MLP. 
[10]. Wykres zależności dokładności od liczby iteracji dla MLP. 
Z wykresu możemy wywnioskować, że im większa liczba iteracji tym większe jest 
prawdopodobieństwo na większa dokładność obliczeniową algorytmu.
Decision Tree
Na działanie tego algorytmu największy wpływ ma parametr określający maksymalna 
liczbę rozgałęzienia węzłów. Na początku w celu stworzenia macierzy parametr ten został 
ustawiony na liczbę 20 przy której dokładność wyniosła 56% . Następnie zbadano wpływ tego 
parametru na dokładność obliczeniową. Zakres parametru dla wykresu to od 2 do 200.
[11]. Tablica pomyłek dla drzewa decyzyjnego. 
[12]. Wykres zależności dokładności od liczby węzłów dla drzewa decyzyjnego
Wykres wskazuje na bardzo niską dokładność obliczeniową algorytmu w zakresie
liczby węzłów 0-10. Natomiast największą dokładność powyżej 70% algorytm otrzymuje dla 
52 węzłów.
Naive Bayes
Dokładność obliczeniowa tego algorytmu wyniosła 52%. Nie posiada on znaczących 
parametrów, które umożliwiałyby adaptację uczenia. Jedyny zmienny parametr to wygładzenie 
danych, które domyślnie zostało włączone.
[13]. Tablica pomyłek dla algorytmu Naiva Bayesa z wygładzaniem. 
Dla porównania po wyłączeniu wygładzenia dokładność algorytmu spadła do 50%.
[14]. Tablica pomyłek dla algorytmu Naiva Bayesa bez wygładzania
7. Podsumowanie
Po wykonaniu badań różnymi wariantami algorytmów, można stwierdzić, że najlepsze
wyniki zwraca nam algorytm Decision Tree – około 70%, zaraz po nim jest algorytm KNN dla 
k =4, który osiąga dokładność równą 67%. Jednak w przypadku tego drugiego, dokładność jego 
obliczeń znacząco spada wraz z zwiększeniem liczby sąsiadów. Zadziwiająco niskie wyniki 
daje nam Perceptron OVR około 34%. Może wynikać to z ilości klas zawartej w bazie danych. 
Algorytm ten działa najlepiej dla baz, które zawierają niską liczbę różnych klas, natomiast baza 
ze szkłem posiada ich aż 7. Dodatkowo baza ta, jest dość trudna do oszacowania ze względu na 
małe różnice w atrybutach pomiędzy różnymi klasami. Algorytm MLP z biblioteki scikit learn, 
wypada stosunkowo dobrze na tle pozostałych algorytmów. Jego dokładność wynosi 54%. 
Jednak jego złożoność obliczeniowa jest skomplikowana dlatego, też potrzebuje dużo czasu na 
wykonanie każdej z iteracji.
