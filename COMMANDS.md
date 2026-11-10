# 📋 Шпаргалка команд для Raspberry Pi & SolarEnergyOptimizer

Быстрый справочник по управлению проектом на **Raspberry Pi 4 B** (и локальном ПК): виртуальное окружение, фоновые сервисы, просмотр логов и безопасное выключение.

---

## 🐍 1. Виртуальное окружение (venv)

Перед запуском любых скриптов обязательно активируй виртуальное окружение.

### Как войти (активировать venv)
Находясь в папке проекта (`SolarEnergyOptimizer`):
```bash
# На Raspberry Pi (Linux):
source .venv/bin/activate
# (или если venv лежит внутри real_time):
source real_time/.venv/bin/activate
```
*На Windows PowerShell:*
```powershell
.\.venv\Scripts\Activate.ps1
```
> Когда venv активен, в начале строки терминала появляется префикс: **`(.venv)`**.

### Как проверить, в каком ты окружении
```bash
which python
# Должно вернуть: .../SolarEnergyOptimizer/.venv/bin/python
```

### Как выйти из venv
```bash
deactivate
```

### Первоначальная установка зависимостей (если создаешь venv с нуля)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r real_time/requirements.txt
```

---

## ⚡ 2. Сервис MQTT (`mqtt_service.py`)

Сервис слушает брокер `cassandra2.tul.cz`, пересчитывает баланс батареи и пишет в базу данных `real_time/data/telemetry.db`.

### Вариант А: Интерактивный запуск (для отладки, вывод прямо в экран)
```bash
# Боевой режим (слушает реальный инвертор)
python3 -m real_time.mqtt_service

# Тестовый режим (с генерацией симулированных данных раз в 2 сек)
python3 -m real_time.mqtt_service --mock --mock-interval 2.0
```
*Чтобы остановить:* нажми **`Ctrl + C`**.

---

### Вариант Б: Фоновый запуск (чтобы работал даже при закрытии SSH)
```bash
nohup python3 -m real_time.mqtt_service > real_time/mqtt.log 2>&1 &
```

### Просмотр логов MQTT
```bash
# Смотреть лог в реальном времени (живой стрим):
tail -f real_time/mqtt.log

# Посмотреть последние 50 строк:
tail -n 50 real_time/mqtt.log

# Посмотреть только ошибки в логе:
grep -i "error" real_time/mqtt.log | tail -n 20
```
*(Чтобы выйти из `tail -f`, нажми **`Ctrl + C`**)*.

### Проверка статуса (запущен ли процесс)
```bash
pgrep -af mqtt_service
```
*(Если выведет номер PID и команду — сервис работает).*

### Остановка фонового сервиса MQTT
```bash
pkill -f mqtt_service
```

---

## 📊 3. Веб-визуализатор Streamlit (`visualizer_realtime.py`)

Дашборд с графиками баланса фаз, ценами OTE ČR и статусом батареи.

### Вариант А: Обычный запуск (для проверки)
```bash
streamlit run real_time/visualizer_realtime.py --server.port 8501 --server.address 0.0.0.0
```
*Чтобы остановить:* нажми **`Ctrl + C`**.

---

### Вариант Б: Фоновый запуск (для постоянной работы)
```bash
nohup streamlit run real_time/visualizer_realtime.py --server.port 8501 --server.address 0.0.0.0 > real_time/visualizer.log 2>&1 &
```

### Как открыть в браузере (с ПК или смартфона)
В браузере введи Tailscale IP малинки:
👉 **`http://100.92.72.6:8501`**

### Просмотр логов визуализатора
```bash
tail -f real_time/visualizer.log
```

### Остановка визуализатора
```bash
pkill -f visualizer_realtime
# или полностью убить все процессы streamlit:
pkill -f streamlit
```

---

## 🚀 4. Быстрый запуск и остановка ВСЕГО сразу

### Способ 1: С помощью готовых скриптов (в 1 команду)
В папку `scripts/` добавлены готовые скрипты:
```bash
# Запустить оба сервиса в фоне:
chmod +x scripts/start_rp.sh scripts/stop_rp.sh   # (только один раз при первом запуске)
./scripts/start_rp.sh

# Остановить оба сервиса:
./scripts/stop_rp.sh
```

---

### Способ 2: Вручную в терминале

#### Запустить оба сервиса в фоне одной строкой:
```bash
nohup python3 -m real_time.mqtt_service > real_time/mqtt.log 2>&1 & nohup streamlit run real_time/visualizer_realtime.py --server.port 8501 --server.address 0.0.0.0 > real_time/visualizer.log 2>&1 &
```

#### Выключить оба сервиса одной командой:
```bash
pkill -f mqtt_service; pkill -f streamlit
```

---

## 🔮 5. Как узнать примерный план алгоритма на завтра

План рассчитывается линейным оптимизатором (LP MPC) на основе спотовых цен OTE ČR и прогноза погоды Open-Meteo.
> [!NOTE]
> Цены OTE ČR на следующий день публикуются **ежедневно в 13:10–13:15 CET**. До 13:15 известна только погода; после 13:15 план рассчитывается на все 24 часа.

### Способ 1: Прямо в веб-дашборде Streamlit
Внизу страницы дашборда на `http://100.92.72.6:8501` открой вкладку:
👉 **`🔮 Zobrazit plán a předpověď arbitráže na zítřek (Spot & FVE)`** — там интерактивно показаны ожидаемая выработка, пиковые часы и действия батареи по часам.

### Способ 2: В терминале через скрипт
```bash
./scripts/plan_tomorrow.sh
# или:
python3 scripts/plan_tomorrow.py
```
*(Выведет красивую сводку: самый дешевый час, пиковый час, выработку солнца и почасовые команды инвертора).*

---

## 🛑 6. Как правильно выключать и перезагружать Raspberry Pi

> [!CAUTION] Никогда не выдергивай провод питания малинки просто так!
> База данных SQLite (`telemetry.db`) работает в режиме высокой производительности (WAL). Внезапное отключение питания может повредить базу данных и файловую систему microSD карты.

### Безопасное выключение:
```bash
sudo poweroff
# или:
sudo shutdown -h now
```
**Что делать дальше:**
1. Соединение SSH закроется.
2. Посмотри на светодиоды на плате Raspberry Pi: зеленый светодиод **ACT** начнет часто мигать (сброс кэша диска), затем погаснет.
3. Подожди **10–15 секунд**, пока останется гореть только стабильный красный светодиод питания **PWR**.
4. **Только теперь можно безопасно отключать кабель питания из розетки.**

### Перезагрузка:
```bash
sudo reboot
```
*(Малинка перезагрузится и через 30–45 секунд снова появится в сети Tailscale).*

---

## 🛠️ 7. Полезные команды диагностики

| Команда | Что делает |
| :--- | :--- |
| `vcgencmd measure_temp` | Температура процессора (норма до 65–70 °C) |
| `free -h` | Использование оперативной памяти (RAM) |
| `df -h /` | Свободное место на карте памяти microSD |
| `tailscale ip -4` | Показать текущий IP адрес в Tailscale |
| `nmcli dev status` | Статус Wi-Fi и проводной сети |
| `ping -c 3 cassandra2.tul.cz` | Проверка доступности брокера TUL |
| `ls -lh real_time/data/telemetry.db` | Размер файла базы данных |
| `htop` | Диспетчер задач (нагрузка CPU и памяти в реальном времени) |
