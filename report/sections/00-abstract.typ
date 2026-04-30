// 00-abstract.typ — Abstract (EN) + Аннотация (RU). Annex 5 §3–4: ≤2000 chars
// each, 5–10 keywords each, bilingual (English body ⇒ Russian abstract required).
// Annex 7 §1.2: the Abstract is NOT numbered and starts on a new page.
// Numbers that depend on final experiments are marked [TODO] and kept out of the
// number-independent prose so they slot in cleanly.

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
comparable. Two method families are contrasted: classical pipelines (mel-
frequency cepstral coefficients with delta features and spectral statistics,
fed to logistic regression, support-vector machines, random forests and gradient
boosting) and deep learning on log-mel spectrograms (a compact convolutional
network and an EfficientNet-B0 transfer model). On heart sounds the best
classical model reaches a mean accuracy of #text(fill: rgb("#b00020"))[[TODO: final MAcc, e.g. 0.90]];
on lung sounds the ICBHI score is #text(fill: rgb("#b00020"))[[TODO: final ICBHI score]].
We additionally analyse how method rankings transfer across the two modalities.
The study is intentionally scoped as honest, leakage-free baselines rather than
state-of-the-art tuning, and a dedicated limitations section delineates what the
results do and do not support clinically.

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
свёрточная сеть и трансферная модель EfficientNet-B0). Лучшее качество на тонах
сердца составляет #text(fill: rgb("#b00020"))[[TODO: итоговое MAcc]];
на дыхательных шумах оценка ICBHI равна
#text(fill: rgb("#b00020"))[[TODO: итоговая оценка ICBHI]]. Отдельно
анализируется перенос ранжирования методов между модальностями.

*Ключевые слова:* аускультация, фонокардиограмма, дыхательный шум, машинное
обучение, глубокое обучение, MFCC, свёрточная нейронная сеть, оценка на уровне
пациента, утечка данных, сравнительное исследование.
