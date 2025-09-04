# ESP32-CSI Tutorial for Linux

Before proceeding with the following steps, ensure that the latest version of Python 3 is installed on your system at `/usr/bin/python3`.

### In the terminal:

```bash
cd ~/Documents
mkdir csi
cd csi
git clone -b v5.5 --recursive https://github.com/espressif/esp-idf.git esp-idf-v5.5      # For version 5.5
# or
git clone --recursive https://github.com/espressif/esp-idf.git        # For version 6.0
git clone --recursive https://github.com/espressif/esp-csi.git
````

In this step, a local directory is created and the ESP-IDF and ESP-CSI repositories are cloned together with their dependencies (`--recursive`).

---

### Setting up ESP-IDF:

```bash
cd esp-idf-v5.5
# or
cd esp-idf
./install.sh
. ./export.sh
```

At this point, the ESP-IDF environment has been successfully initialized.

---

### Connecting the ESP32 to a USB Port

Identify the port to which the device is connected (e.g., `/dev/ttyUSB[X]`, where `[X]` corresponds to the USB device number):

```bash
ls /dev/tty*
# or
lsusb
# or
dmesg | grep *ttyUSB*
```
---

Additional details can be found in [esp\_csi\_tool.py](https://github.com/espressif/esp-csi/blob/master/examples/esp-radar/console_test/README.md).

### Configuring the ESP32-CSI Firmware

Navigate to the example folder: `examples/esp-radar/console_test`

```bash
cd ~/Documents/csi/esp-csi/examples/esp-radar/console_test
idf.py set-target esp32    # Defines the correct firmware target; verify with `esptool.py flash_id` (e.g., esp32, esp32s3, esp32c2, etc.)
idf.py menuconfig          # Provides access to ESP32 configuration settings
```

The UART console baud rate must be modified as shown in the image below:

<img width="1909" height="991" alt="menuconfig_change" src="https://github.com/user-attachments/assets/0592bd2c-6eb9-4583-944d-347a714d77e2" />

After applying the changes in `menuconfig`, execute the following commands:

```bash
idf.py build               # Compiles and links the project into a binary file for the ESP32
                           # Up to this step, the ESP32 does not need to be connected to the USB port.
                           # From the next step onwards, a USB connection is required.
idf.py -p /dev/ttyUSB[X] flash -b 921600      # Uploads the compiled binary to the ESP32
```

At this point, the **`csi console_test`** firmware is running on the ESP32.

To verify its execution, run the monitor:

```bash
idf.py -p /dev/ttyUSB[X] monitor -b 921600
```

Once the interactive prompt (`csi>`) appears, press *Enter* and execute:

```bash
wifi_config -s <SSID> -p <Password>        # Note: Both SSID and password are case-sensitive
```

An output similar to the following should be displayed:

<img width="1909" height="991" alt="port_monitor" src="https://github.com/user-attachments/assets/3d0bca98-6c22-430f-9e25-1e48b31660e5" />

To exit the application, use `Ctrl + ]` or the sequence `Ctrl + T` followed by `Ctrl + X`.

---

### Executing `esp_csi_tool`

```bash
cd ~/Documents/csi/esp-csi/examples/esp-radar/console_test/tools
pip install pandas PyQt5 pyqtgraph scipy  # Install required Python dependencies
```

Once all dependencies are installed, run:

```bash
python esp_csi_tool.py -p /dev/ttyUSB[X]
```

Enter the SSID and password of your wireless network.
**Important:** the SSID is case-sensitive.

---

The collected CSI data will be stored in the following directory:

```bash
~/Documents/csi/esp-csi/examples/esp-radar/console_test/tools/log
```

This folder contains **three `.csv` files** and **one `log_data.txt`** file with detailed CSI information.

