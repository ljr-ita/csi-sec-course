# ESP32-CSI Tutorial for Linux

Before following the next steps, make sure you have the latest Python 3 version installed on your system at `/usr/bin/python3`.

### In the terminal:

```
cd ~/Documents
mkdir csi
cd csi
git clone -b v5.5 --recursive https://github.com/espressif/esp-idf.git esp-idf-v5.5      #For version 5.5
# or
git clone --recursive https://github.com/espressif/esp-idf.git        #For version 6.0
git clone --recursive https://github.com/espressif/esp-csi.git
```
In the step above, you already created a local path and downloaded ESP-IDF and ESP-CSI with all their dependencies (`recursive`)

### Set up ESP-IDF:

```
cd esp-idf-v5.5
# or
cd esp-idf
./install.sh
. ./export.sh
```

You are now in the ESP-IDF environment.

---

### Connect your ESP32 to a USB port

Check which port the device is attached to (look for `/dev/ttyUSB[X]`):      (`X` is the number of USB device)

```
ls /dev/tty*
# or
lsusb
# or
dmesg | grep *ttyUSB*
```

---
The topics below can to be access in [esp_csi_tool.py](https://github.com/espressif/esp-csi/blob/master/examples/esp-radar/console_test/README.md)

### Set up your ESP32 firmware

Change to the example folder: `examples/esp-radar/console_test`

```
cd ~/Documents/csi/esp-csi/examples/esp-radar/console_test
idf.py set-target esp32    #It's used to set the right firmware, check it runing `esptool.py flash_id` (esp32, esp32s3, esp32c2,...)
idf.py menuconfig          #It's used to visualize the settings for the ESP32
```
You must change the UART console baud rate as shown in the image below:
<img width="1909" height="991" alt="menuconfig_change" src="https://github.com/user-attachments/assets/0592bd2c-6eb9-4583-944d-347a714d77e2" />

After you make the change in menuconfig, follow the next instructions:
```
idf.py build               #It compiles and links your project into a binary file for the ESP32
#Until step above, you don't need have your ESP32 connect to USB, but to step below you need do it
idf.py -p /dev/ttyUSB[X] flash -b 921600      #It's used to upload the compiled code to the ESP32 chip
```
The firmware **`csi console_test`** is now running on the ESP32.

Run the `monitor` to check if the firmware is working:
```
idf.py -p /dev/ttyUSB[X] monitor -b 921600
```
When an interactive prompt like csi> appears, press Enter and run the command:
```
wifi_config -s <SSID> -p <Password>        #Be atention, it's case sensitive
```
Something like that will appear:
<img width="1909" height="991" alt="port_monitor" src="https://github.com/user-attachments/assets/3d0bca98-6c22-430f-9e25-1e48b31660e5" />

`Ctrl + ]` or `Ctrl + T` followed by `Ctrl + X`, like `Ctrl + T and Ctrl + X` for exit the app.

---
### Running `esp_csi_tool`

```
cd ~/Documents/csi/esp-csi/examples/esp-radar/console_test/tools
pip install pandas PyQt5 pyqtgraph scipy  # Install required Python dependencies
```

After installing all libraries:

```
python esp_csi_tool.py -p /dev/ttyUSB[X]
```

Enter the SSID and password of your wireless network. **Be careful: SSID is case-sensitive!**

---
The CSI data will be recorded in the folder:  
```
~/Documents/csi/esp-csi/examples/esp-radar/console_test/tools/log
```

This folder contains **three `.csv` files** and **one `log_data.txt`** file with the CSI information.


