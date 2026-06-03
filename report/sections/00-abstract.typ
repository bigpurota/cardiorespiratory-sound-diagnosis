
#heading(numbering: none, outlined: true)[Annotation]

This work presents a reproducible comparative study of machine-learning methods
for the acoustic screening of cardiorespiratory pathologies from auscultation sound.
Two modalities are examined under one unified, leakage-safe pipeline: heart sounds
(phonocardiograms) from PhysioNet/CinC 2016 and respiratory sounds from ICBHI 2017.
A third modality, arterial bruits, is treated analytically, as no public dataset
suitable for supervised learning exists. The central methodological contribution is
an identical evaluation protocol for both modalities: strictly grouped
partitioning (patient-level for lung, recording-level for heart) that prevents the
window-level leakage inflating many results, and a
shared balanced metric, the mean accuracy
$"MAcc" = ("Se" + "Sp") \/ 2$, that makes the tasks directly comparable. Two method families
are contrasted: classical pipelines (MFCC with delta and spectral features, fed to
logistic regression, support-vector machines, random forests and gradient boosting)
and deep learning on log-mel spectrograms (a compact CNN and an EfficientNet-B0
transfer model, both tuned by hyperparameter optimisation (HPO) over $128$ trials and
reported as mean±std across three seeds). On heart sounds the best classical model
(XGBoost, extended features) reaches $"MAcc" = 0.903$ ($"Se" = 0.946$, $"Sp" = 0.859$); the
tuned EfficientNet-B0 reaches $0.898 plus.minus 0.008$, comparable but not ahead of it.
On lung sounds all methods occupy one performance tier ($"ICBHI" approx 0.54"–"0.56$),
with EfficientNet-B0 marginally best at $0.555 plus.minus 0.016$. The cross-modal analysis
reveals asymmetric transfer: lung-pretrained features carry over strongly to heart,
whereas heart→lung transfer is near-neutral, and a joint multi-task model roughly
preserves both. An Audio Spectrogram Transformer is integrated and code-verified,
with full fine-tuning left to future work. The study is scoped as honest,
leakage-free baselines, and a limitations section sets out what the results do
and do not clinically support.

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
артериальные шумы — рассматривается аналитически, так как общедоступного
размеченного набора для обучения с учителем не существует. Ключевой методический
вклад — единый протокол оценки для обеих модальностей: строгое групповое
разбиение, исключающее утечку данных (на уровне пациента для лёгких и на уровне
записи для сердца, поскольку CinC 2016 не публикует сопоставление записей с
пациентами), фиксированные сиды, закреплённые версии библиотек и единое семейство
метрик ($("Se" + "Sp") \/ 2$), делающее задачи напрямую сопоставимыми. Сравниваются
классические конвейеры (мел-кепстральные коэффициенты с дельта-признаками и
спектральными статистиками, подаваемые в логистическую регрессию, метод опорных
векторов, случайный лес и градиентный бустинг) и глубокое обучение на лог-мел
спектрограммах (компактная свёрточная сеть и трансферная модель EfficientNet-B0,
настроенные посредством HPO и усреднённые по трём сидам). На тонах сердца лучший
результат — $"MAcc" = 0.903$ (XGBoost, $"Se" = 0.946$, $"Sp" = 0.859$); EfficientNet-B0
достигает $0.898 plus.minus 0.008$ — сопоставимо, но классическая модель остаётся численно
лучшей. На дыхательных шумах все методы находятся в одном диапазоне
($"ICBHI" approx 0.54"–"0.56$), лучший — $0.555 plus.minus 0.016$ (EfficientNet-B0). Межмодальный анализ
выявляет асимметричный перенос признаков: модели, предобученные на лёгких, хорошо
переносятся на сердце, обратный перенос нейтрален; многозадачная модель сохраняет
обе модальности. Интеграция Audio Spectrogram Transformer верифицирована;
полноценное дообучение вынесено в перспективы.

#heading(level: 2, numbering: none, outlined: false)[Ключевые слова]
аускультация, фонокардиограмма, дыхательный шум, машинное
обучение, глубокое обучение, MFCC, свёрточная нейронная сеть, групповое
разбиение, утечка данных, сравнительное исследование.
