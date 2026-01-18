
Учебный проект по дополнению текстов. В нем обучена нейросеть, которая на основе начала фразы предсказывает её продолжение.

Структура проекта:

text-autocomplete/
├── data/                            # Датасеты
│   ├── raw_dataset.csv              # "сырой" скачанный датасет
│   └── dataset_processed.csv        # "очищенный" датасет
│   ├── train.csv                    # тренировочная выборка
│   ├── val.csv                      # валидационная выборка
│   └── test.csv                     # тестовая выборка
│
├── src/                             # Весь код проекта
│   ├── data_utils.py                # Обработка датасета
|   ├── next_token_dataset.py        # код с torch Dataset'ом 
│   ├── lstm_model.py                # код lstm модели
|   ├── eval_lstm.py                 # замер метрик lstm модели
|   ├── lstm_train.py                # код обучения модели
|   ├── eval_transformer_pipeline.py # код с запуском и замером качества трансформера
│
├── models/                          # веса обученных моделей
|
├── solution.ipynb                   # ноутбук с решением
└── requirements.txt                 # зависимости проекта 
└── README.md                        # описание и выводы

Датасеты и веса созданы, но не добавлены в удаленный репозиторий, так как по заданию он должен быть менее 10 Мб.

Из-за нехватки времени первая версия обучена только на первых 20000 строк. На выделенной виртуалке одна эпоха обучается около часа, а потом где то в конце падает с 'OutOfMemoryError'. Переобучиться с небольшим значением батча (что должно победить OutOfMemoryError) не успел к сроку, но сделаю во вторую итерацию.

#########################
#########################
Выводы 
#########################
#########################
Сами метрики очень низкие, хотя в примерах допонения осмысленные (преимущественно). Из-за нехватки времени первая версия обучена только на первых 20000 строк. Как только обучение на полном наборе завершится, метрики должны возрасти.

По метрикам пока на валидации модели сопоставимы: LSTM чуть лучше по ROUGE‑1 (0.0673 vs 0.0655), а distilgpt2 немного лучше по ROUGE‑2 (0.0061 vs 0.0053), то есть различия небольшие. ROUGE оценивает совпадение n‑грамм и не всегда отражает связность текста при открытой генерации.
​
По примерам distilgpt2 заметно чаще генерирует связные и осмысленные продолжения, тогда как LSTM нередко выдаёт повторяющиеся “частотные” слова и менее читаемый текст. Поэтому для продукта лучше рекомендовать distilgpt2 как более качественную “из коробки” модель, а LSTM — как вариант только при жёстких ограничениях по ресурсам и готовности дополнительно тюнить обучение и генерацию.
​

#########################
Зафиксирую метрики и примеры, которые получились в последней актуальной версии:
#########################
Epoch 1/5 [train]: 100%|██████████| 63/63 [05:16<00:00, 5.02s/it]
Epoch 1/5 [val]: 100%|██████████| 8/8 [00:14<00:00, 1.87s/it]
Epoch 1: Train Loss=8.7930, Val Loss=7.3892, ROUGE1-F=0.0478, ROUGE2-F=0.0000 (n=1920)
Saved best checkpoint to: models\lstm_best.pt (val_loss=7.3892)

Epoch 2/5 [train]: 100%|██████████| 63/63 [05:25<00:00, 5.16s/it]
Epoch 2/5 [val]: 100%|██████████| 8/8 [00:15<00:00, 1.92s/it]

Epoch 2: Train Loss=7.2491, Val Loss=7.2348, ROUGE1-F=0.0267, ROUGE2-F=0.0033 (n=1920)
Saved best checkpoint to: models\lstm_best.pt (val_loss=7.2348)

Epoch 3/5 [train]: 100%|██████████| 63/63 [05:05<00:00, 4.85s/it]
Epoch 3/5 [val]: 100%|██████████| 8/8 [00:14<00:00, 1.86s/it]

Epoch 3: Train Loss=7.0847, Val Loss=7.0943, ROUGE1-F=0.0332, ROUGE2-F=0.0036 (n=1920)
Saved best checkpoint to: models\lstm_best.pt (val_loss=7.0943)

Epoch 4/5 [train]: 100%|██████████| 63/63 [05:03<00:00, 4.81s/it]
Epoch 4/5 [val]: 100%|██████████| 8/8 [00:15<00:00, 1.96s/it]

Epoch 4: Train Loss=6.9298, Val Loss=6.9561, ROUGE1-F=0.0649, ROUGE2-F=0.0050 (n=1920)
Saved best checkpoint to: models\lstm_best.pt (val_loss=6.9561)

Epoch 5/5 [train]: 100%|██████████| 63/63 [05:06<00:00, 4.86s/it]
Epoch 5/5 [val]: 100%|██████████| 8/8 [00:14<00:00, 1.87s/it]

Epoch 5: Train Loss=6.7795, Val Loss=6.8307, ROUGE1-F=0.0673, ROUGE2-F=0.0053 (n=1920)
Saved best checkpoint to: models\lstm_best.pt (val_loss=6.8307)
LSTM VAL: {'rouge1_f': 0.06725882760349697, 'rouge2_f': 0.005282486610611611, 'n_samples': 1920}
GPT2 VAL: {'rouge1_f': 0.06550649469167424, 'rouge2_f': 0.006085652538621535, 'n_samples': 1903}

PREFIX: i am going
LSTM +: to it in the time is of up one 3 we it is in work awww to work for
GPT2 +: to be.”

PREFIX: tomorrow i will
LSTM +: the you i was from the morning good not was but it i and now i so he but i
GPT2 +: keep this for the long term.
But if you have any questions, then just write this:
PREFIX: this movie is
LSTM +: in the i was out i have i don t am was to the and of s day no i
GPT2 +: a pretty good one as I haven't heard much about it before.) I'm very impressed with this
PREFIX: Company Google is
LSTM +: it is my now it is too it be good day is so i miss you could feel i think
GPT2 +: making changes to its search engine and search engine to replace its search engine for the Google Search engine and
PREFIX: If you compare Google and Yandex, you could say that
LSTM +: of my bad at work not have to get to this in twitter i would it 2 all back to
GPT2 +: Yandex uses the same techniques as Google, which relies on artificial intelligence to make the pages for
PREFIX: Our mentor is very smart and
...
PREFIX: I would like
LSTM +: for the time on for in the that to the i t go too me it s the i would
GPT2 +: to thank Mr. B. Kallerman for his thoughtful work, particularly his outstanding contributions, the