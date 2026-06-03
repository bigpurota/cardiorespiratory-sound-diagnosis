
#heading(numbering: none, outlined: true)[Annotation]

This work presents a reproducible comparative study of machine-learning methods
for the acoustic screening of cardiorespiratory pathologies from auscultation sound.
Two modalities are examined under one unified, leakage-safe pipeline: heart sounds
(phonocardiograms) from PhysioNet/CinC 2016 and respiratory sounds from ICBHI 2017.
A third modality, arterial bruits, is treated analytically, as no public dataset
suitable for supervised learning exists. The central methodological contribution is
a single shared evaluation protocol for both modalities: strictly grouped
partitioning (patient-level for lung, recording-level for heart) preventing
window-level leakage, and a shared balanced metric, the mean accuracy
$"MAcc" = ("Se" + "Sp") \/ 2$, that makes the tasks directly comparable. Two families are
contrasted: classical pipelines (MFCC with delta and spectral features fed to four
standard classifiers) and deep learning on log-mel spectrograms (a compact CNN and an
EfficientNet-B0, tuned by hyperparameter optimisation over $128$ trials and reported as
mean±std across three seeds). On heart sounds the best classical model
(XGBoost, extended features) reaches $"MAcc" = 0.903$ ($"Se" = 0.946$, $"Sp" = 0.859$); the
tuned EfficientNet-B0 reaches $0.898 plus.minus 0.008$, comparable but not ahead of it.
On lung sounds all methods occupy one tier; a fine-tuned Audio Spectrogram Transformer
is the single best ($"ICBHI" = 0.594 plus.minus 0.023$, three seeds), above EfficientNet-B0
($0.555 plus.minus 0.016$). A seed-robust cross-modal analysis finds that classical method
rankings do not transfer between modalities, whereas deep encoders transfer benignly in
both directions; a transfer asymmetry seen at a single seed did not survive multi-seed
averaging. On the CirCor DigiScope 2022 dataset, with a patient-level split, deep
learning overtakes the classical models on murmur detection ($0.810$ vs. $0.688$),
reversing the CinC ordering. A limitations section bounds the clinical claims.

#heading(level: 2, numbering: none, outlined: false)[Keywords]
auscultation, phonocardiogram, respiratory sound, machine learning,
deep learning, MFCC, convolutional neural network, grouped-split evaluation,
data leakage, comparative study.

#v(0.6em)
#heading(numbering: none, outlined: true)[Аннотация]

В работе представлено воспроизводимое сравнительное исследование методов
машинного обучения для акустического скрининга патологий кардиореспираторной
системы по звукам аускультации. В рамках единого, устойчивого к утечке данных
конвейера рассматриваются две модальности: тоны сердца (фонокардиограммы) из
PhysioNet/CinC 2016 и дыхательные шумы из ICBHI 2017. Третья модальность —
артериальные шумы — рассмотрена аналитически: открытого размеченного набора
для обучения с учителем не существует. Ключевой методический
вклад — единый протокол оценки для обеих модальностей: строгое групповое
разбиение, исключающее утечку данных (на уровне пациента для лёгких и записи для
сердца), фиксированные начальные значения генераторов, закреплённые версии библиотек и единое семейство
метрик ($("Se" + "Sp") \/ 2$), делающее задачи напрямую сопоставимыми. Сравниваются
классические конвейеры (мел-кепстральные коэффициенты с дельта-признаками и
спектральными статистиками, поданные в четыре стандартных классификатора) и
глубокое обучение на лог-мел
спектрограммах (компактная свёрточная сеть и трансферная модель EfficientNet-B0,
настроенные оптимизацией гиперпараметров и усреднённые по трём прогонам с разной
инициализацией). На тонах сердца лучший
результат — $"MAcc" = 0.903$ (XGBoost, $"Se" = 0.946$, $"Sp" = 0.859$); EfficientNet-B0
достигает $0.898 plus.minus 0.008$ — сопоставимо, но классическая модель остаётся численно
лучшей. На дыхательных шумах все методы находятся в одном диапазоне
($"ICBHI" approx 0.54"–"0.59$); лучший результат — у дообученного Audio
Spectrogram Transformer ($0.594 plus.minus 0.023$), выше EfficientNet-B0 ($0.555 plus.minus 0.016$).
Межмодальный анализ по трём прогонам показывает: ранги классических методов между
модальностями не переносятся, тогда как глубокие энкодеры переносятся доброкачественно
в обе стороны (асимметрия одного прогона не пережила усреднение). На датасете
CirCor DigiScope 2022 с разбиением по пациентам глубокое обучение обгоняет классику
в детекции шумов ($0.810$ против $0.688$), обращая порядок CinC.

#heading(level: 2, numbering: none, outlined: false)[Ключевые слова]
аускультация, фонокардиограмма, дыхательный шум, машинное
обучение, глубокое обучение, MFCC, свёрточная нейронная сеть, групповое
разбиение, утечка данных, сравнительное исследование.
