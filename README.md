# BipedalWalker-v3 - sterowanie dwunożnym robotem (Gymnasium) :running: :robot:
Celem projektu było wytrenowanie agenta do poruszania się w wirtualnym środowisku, wykorzystując różne metody uczenia/sterowania.
Użyłam biblioteki Gymnasium oraz jednej z jej gotowych gier - Bipedal Walker, polegającej na poruszaniu się robotem na dwóch nogach po lekko nierównym terenie.

Robot posiada 4 stawy - biodrowy i kolanowy dla każdej nogi.

## Simple
Pierwszą metodą był bardzo prosty skrypt sterujący robotem: jeśli "kadłub" odchyla się do tyłu, stara się ruszać do przodu, jeśli kadłub odchyla się w przód, stara się balansować przesuwając się do tyłu. Rozwiązanie to nie działa dobrze, robot porusza się chaotycznie i zwykle przewraca się, w najlepszym wypadku wykonuje jeden krok. Nie eksperymentowałam jednak dalej z tym podejściem, ponieważ wolałam skupić się na metodach wykorzystujących machine learning.

![Demo1](/images/bipedal_simple.gif)

## Reinforcement Learning - PPO
Świetnie w tym problemie sprawdził się reinforcement learning (uczenie przez wzmacnianie). Do uczenia agenta wykorzystałam metodę PPO (Proximal Policy Optimization) z biblioteki stable_baselines3.

![Demo2](/images/bipedal_PPO_v1.gif)

timesteps=100_000

Nauka dla 10 tys. kroków wykonywała się około 15 minut, po tym czasie agent wytrenował się wystarczająco, aby poruszać się powoli, ale nie przewracając się. Przyjął strategię poruszania się "w rozkroku", dla zachowania równowagi.

![Demo3](/images/bipedal_PPO_v2.gif)

timesteps=1_000_000\
Nauka dla 1 miliona kroków trwała prawie 2 godziny. Agent wytrenował się na tyle dobrze, że bez problemu szybko pokonuje teren. Prawdopodobnie wydłużając ten trening, agent nauczył by się chodzić / biegać jeszcze sprawniej.

## Algorytm genetyczny
Pierwszy eksperyment przeprowadziłam tworząc 40 pokoleń, każde o populacji równej 50.\
![Demo4](/images/learning_process_GA.png)
![Demo4](/images/best_agent_performance.gif)

Najlepszy agent osiągnął wynik zaledwie -82 punktów, dlatego przewraca się dosyć szybko, widać jednak próbę wykonania długiego kroku, aby poruszyć się jak najdalej.

Drugi eksperyment przeprowadziłam tworząc 40 pokoleń, każde o populacji równej 100.\
![Demo4](/images/learning_process_GA_2.png)
![Demo4](/images/new_best_agent_performance.gif)\
W tym przypadku agent poradził sobie już dużo lepiej, umie pokonać już znaczny dystans, jednak dość powoli, "czołgając" się na jednej nodze.

Zaobserwowałam, że pokolenia, które otrzymywały wyniki średnie (0-70 punktów) wykonywały się dłużej, niż pokolenia z bardzo słabymi lub dobrymi wynikami. Wywnioskowałam, że wynika to z przyjętej strategii poruszania się robota: jeśli osiąga słaby wynik, znaczy to, że przewraca się po kilku krokach (symulacja kończy się), natomiast jeśli osiąga dobry wynik, szybko dociera do końca terenu (symulacja również się kończy). Jeśli jednak ma wynik średni, prawdopodobnie porusza się bardzo wolno, ale się nie przewraca, przez co symulacja kończy się dopiero, gdy osiągnie dany limit czasowy (w moim programie dla jednego episode przyjęłam timesteps = 2000, aby symulacja kończyła się, jeśli agent porusza się zbyt wolno lub wcale).

## Fuzzy
![Demo3](/images/bipedal_fuzzy.gif)