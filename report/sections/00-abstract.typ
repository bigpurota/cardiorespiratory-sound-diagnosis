// 00-abstract.typ — Abstract (EN) + Аннотация (RU). Annex 5 §3–4: ≤2000 chars
// each, 5–10 keywords each, bilingual (English body ⇒ Russian abstract required).
// Annex 7 §1.2: the Abstract is NOT numbered and starts on a new page.
// All numbers final (HPO+multi-seed DL + cross-modal). Updated 2026-06-02.

// Unnumbered headings: use `heading(... numbering: none)` equivalent via [#heading].
#heading(numbering: none, outlined: true)[Abstract]

This work presents a reproducible comparative study of machine-learning methods
for diagnosing cardiorespiratory pathologies from auscultation sound. Two
clinical modalities are examined under one unified, leakage-safe pipeline: heart
sounds (phonocardiograms) from the PhysioNet/CinC 2016 database and respiratory
sounds from the ICBHI 2017 database. A third modality — arterial bruits — is
treated analytically, because no public dataset suitable for supervised learning
currently exists. The central methodological contribution is the application of
an identical evaluation protocol to both modalities: strictly patient-level
train/test partitioning that prevents the data leakage which inflates many
published results, fixed random seeds, pinned software versions, and a single
shared metric family, ((Se + Sp) / 2), that makes the two tasks directly
comparable. Two method families are contrasted: classical pipelines (mel-frequency cepstral
coefficients with delta features and spectral statistics, fed to logistic regression,
support-vector machines, random forests and gradient boosting) and deep learning on
log-mel spectrograms (a compact convolutional network and an EfficientNet-B0
transfer model, both HPO-tuned across 128 trials and reported as mean±std over
three seeds). On heart sounds the best classical model (XGBoost with extended
spectral features) reaches MAcc = 0.903 (Se = 0.946, Sp = 0.859); the tuned
EfficientNet-B0 reaches MAcc = 0.898 ± 0.008, closing the deep-vs-classical gap
to within noise. On lung sounds all methods occupy the same performance tier
(ICBHI ≈ 0.54–0.56), with EfficientNet-B0 marginally best at 0.555 ± 0.016.
The cross-modal analysis reveals asymmetric transfer: lung-pretrained features
carry over strongly to heart, but heart→lung transfer is near-neutral; a joint
multi-task model roughly preserves both modalities. An Audio Spectrogram Transformer
was integrated and verified but not fully evaluated within the project's time budget.
The study is intentionally scoped as honest, leakage-free baselines and a dedicated
limitations section delineates what the results do and do not support clinically.

*Keywords:* auscultation, phonocardiogram, respiratory sound, machine learning,
deep learning, MFCC, convolutional neural network, patient-level evaluation,
data leakage, comparative study.

#v(0.6em)
#heading(numbering: none, outlined: true)[Аннотация]

В работе представлено воспроизводимое сравнительное исследование методов
машинного обучения для диагностики патологий кардиореспираторной системы по
звукам аускультации. В рамках единого, устойчивого к утечке данных конвейера
рассматриваются две клинические модальности: тоны сердца (фонокардиограммы) из
базы PhysioNet/CinC 2016 и дыхательные шумы из базы ICBHI 2017. Третья
модальность — артериальные шумы — рассматривается аналитически, поскольку
общедоступного размеченного набора данных для обучения с учителем в настоящее
время не существует. Ключевой методический вклад состоит в применении
идентичного протокола оценки к обеим модальностям: строгого разбиения на обучение
и тест на уровне пациента, исключающего утечку данных, фиксированных начальных
значений генераторов случайных чисел, закреплённых версий библиотек и единого
семейства метрик ((Se + Sp) / 2), делающего обе задачи напрямую сопоставимыми.
Сравниваются два семейства методов: классические конвейеры (мел-частотные
кепстральные коэффициенты с дельта-признаками и спектральными статистиками,
подаваемые в логистическую регрессию, метод опорных векторов, случайный лес и
градиентный бустинг) и глубокое обучение на лог-мел спектрограммах (компактная
свёрточная сеть и трансферная модель EfficientNet-B0, настроенные посредством HPO
и оценённые как среднее±стд по трём случайным сидам). На тонах сердца лучший
классический результат — MAcc = 0.903 (XGBoost, Se = 0.946, Sp = 0.859); настроенный
EfficientNet-B0 достигает MAcc = 0.898 ± 0.008, сводя разрыв к статистическому шуму.
На дыхательных шумах все методы находятся в одном диапазоне качества
(ICBHI ≈ 0.54–0.56); лучший результат — 0.555 ± 0.016 (EfficientNet-B0).
Межмодальный анализ выявляет асимметричный перенос признаков: предобученные на
лёгких модели хорошо переносятся на сердце, но обратный перенос нейтрален.
Совместная многозадачная модель сохраняет качество обеих модальностей.
Дополнительно верифицирована интеграция Audio Spectrogram Transformer;
полноценная дообученная оценка вынесена в перспективы.

*Ключевые слова:* аускультация, фонокардиограмма, дыхательный шум, машинное
обучение, глубокое обучение, MFCC, свёрточная нейронная сеть, оценка на уровне
пациента, утечка данных, сравнительное исследование.
