# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #5 выполнил:
- Сажин Егор Алексеевич
- РИ210946
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Интеграция экономической системы в проект Unity и обучение ML-Agent

## Задание 1
### Измените параметры файла. yaml-агента и определить какие параметры и как влияют на обучение модели.

Изначальные данные и результаты:

```py
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```
![Снимок экрана 2022-11-29 143837](https://user-images.githubusercontent.com/102538132/204494781-e6f67121-1753-4e24-9805-551714c01211.png)
![данные](https://user-images.githubusercontent.com/102538132/204512669-c98d0b34-e69e-4af0-abb9-55140ba90e7c.png)
![beta](https://user-images.githubusercontent.com/102538132/204516830-05876f9b-ac4a-439e-beb2-3d40481d8f6c.png)


1. Изменим epsilon c 0.2 на 1.2:


```py
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 1.2 ###
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10

```

Получаются такие графики:


![данные](https://user-images.githubusercontent.com/102538132/204514150-dd8c2fce-36bf-42cb-95cf-c6e36ffb4fb0.png)
![beta](https://user-images.githubusercontent.com/102538132/204514808-9c5383b5-4fa1-4e93-8541-c0a60a6b2705.png)


2. Изменим buffer_size c 10240 на 150:
```py
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 150 ###
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10

```


Получаются такие графики:


![текущ_знач](https://user-images.githubusercontent.com/102538132/204511466-e1a3da3b-f15f-43fe-be3c-066c9c4cd793.png)
![beta](https://user-images.githubusercontent.com/102538132/204515819-b5326f52-aa77-4b60-b2b0-35a24672ce55.png)



3. Изменим batch_size c 1024 на 4096:


```py
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 4096
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```


Получаются такие графики:

![data](https://user-images.githubusercontent.com/102538132/204518010-483b9768-a337-4401-96cd-2fb61099ebd7.png)
![beta](https://user-images.githubusercontent.com/102538132/204518020-153ce0b2-79ea-4dc3-8cf1-8d1aa207d437.png)


4. Изменим lambd c 0.95 на 0.1:


```py
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.1 ###
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```


Получаются такие графики:


![data](https://user-images.githubusercontent.com/102538132/204520149-87fa8e16-9eef-479f-8f20-db5e197a8fc4.png)
![beta](https://user-images.githubusercontent.com/102538132/204520162-434a851a-9e2a-457f-93b6-027be77b7900.png)


5. Изменим num_layers c 2 на 100:
```py
behaviors:
  Economic:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 100
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```


Получаются такие графики:


![data](https://user-images.githubusercontent.com/102538132/204522047-9c4dc5e9-7892-4924-baab-91b3f81ebcf5.png)
![beta](https://user-images.githubusercontent.com/102538132/204522059-247bf24c-90c8-4d55-b186-95aca2f9c0e9.png)


Вывод: при изменении всех перечисленных значений графики меняются незначительно либо вовсе не меняются.Также при изменении параметра buffer_size график "Value Loss" отличается от первоначального графика. Но также можно заметить, что при изменении параметра num_layers, график меняется значительно относительно первоначальных графиков и обучение также по времени становится дольше. В общем можно сказать, что изменение текущих(не всех.исключение: параметр num_players) данных параметров положительно сказывается на обучении модели Economic.


## Задание 2
### Опишите результаты, выведенные в TensorBoard.

Результаты: если взять информацию за основу, графики у первоначального файла Yaml Economic: график "Cumulative Reward" при хорошем обучении должен возрастать, также он не должен превышать 1 по вертикали; график "Episode Length" на скринах это плохо видно, но он тоже должен возрастать; график "Policy Loss" здесь тоже не понятно, но в целом он возрастает; и последний график "Value Loss" убывает. На основе взятых 5 параметров epsilon, buffer_size, batch_size, lambd и num_players и при изменении их, графики (у 4 параметров) не изменились, заметим только то, что графики у параметра num_players сильно отличаются. Нельзя сказать, что в плохую сторону у параметра num_players, так как по графикам обучение идет хорошо, но по времении обучение занимет больше времени, чем у других параметров. 


## Выводы
В ходе выполнения данной лабораторной работы, я научился интегрировать экономическую систему в проект Unity в связке с MLAgent, наблюдать за изменениями при изменении параметров в Yaml файле, делать графики на основе результатов обучения MLAgent и анализировать графики.


| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
