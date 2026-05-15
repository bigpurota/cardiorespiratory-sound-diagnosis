// 00-abstract.typ — Abstract (EN) + Аннотация (RU). Annex 5 §3–4: ≤2000 chars
// each, 5–10 keywords each, bilingual (English body ⇒ Russian abstract required).
// Annex 7 §1.2: the Abstract is NOT numbered and starts on a new page.
// All numbers final (HPO+multi-seed DL + cross-modal). Updated 2026-06-02.

// Unnumbered headings: use `heading(... numbering: none)` equivalent via [#heading].
#heading(numbering: none, outlined: true)[Abstract]

This work presents a reproducible comparative study of machine-learning methods
for diagnosing cardiorespiratory pathologies from auscultation sound. Two
modalities are examined under one unified, leakage-safe pipeline: heart sounds
(phonocardiograms) from PhysioNet/CinC 2016 and respiratory sounds from ICBHI 2017.
A third modality — arterial bruits — is treated analytically, as no public dataset
suitable for supervised learning exists. The central methodological contribution is
an identical evaluation protocol for both modalities: strictly grouped, leakage-safe
partitioning that prevents the leakage that inflates many published results, fixed
seeds, pinned software, and a shared balanced metric, mean accuracy
MAcc = (Se + Sp) / 2, that makes the tasks directly comparable. Two method families
are contrasted: classical pipelines (MFCC with delta and spectral features, fed to
logistic regression, support-vector machines, random forests and gradient boosting)
and deep learning on log-mel spectrograms (a compact CNN and an EfficientNet-B0
transfer model, both tuned by hyperparameter optimisation (HPO) over 128 trials and
reported as mean±std across three seeds). On heart sounds the best classical model
(XGBoost, extended features) reaches MAcc = 0.903 (Se = 0.946, Sp = 0.859); the
tuned EfficientNet-B0 reaches 0.898 ± 0.008, closing the deep-versus-classical gap
to within noise. On lung sounds all methods occupy one performance tier
(ICBHI ≈ 0.54–0.56), EfficientNet-B0 marginally best at 0.555 ± 0.016. The
cross-modal analysis reveals asymmetric transfer: lung-pretrained features carry
over strongly to heart, but heart→lung transfer is near-neutral; a joint multi-task
model roughly preserves both. An Audio Spectrogram Transformer was integrated and
verified but not fully fine-tuned within the time budget. The study is deliberately
scoped as honest, leakage-free baselines, with a limitations section delineating
what the results do and do not clinically support.

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
общедоступного размеченного набора данных для обучения с учителем не существует. Ключевой методический вклад состоит в применении
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
Интеграция Audio Spectrogram Transformer верифицирована; полноценная дообученная
оценка вынесена в перспективы.

*Ключевые слова:* аускультация, фонокардиограмма, дыхательный шум, машинное
обучение, глубокое обучение, MFCC, свёрточная нейронная сеть, оценка на уровне
пациента, утечка данных, сравнительное исследование.
