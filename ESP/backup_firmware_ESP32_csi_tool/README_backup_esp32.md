
To successfully follow this tutorial, it is required that the development environment is already configured with [ESP-IDF v5.x](https://github.com/espressif/esp-idf.git) and the [ESP-CSI](https://github.com/espressif/esp-csi.git) framework properly installed.

```
git clone --branch release/v5.5 https://github.com/espressif/esp-idf.git
```

**Firmware Backup and Recovery Procedure for ESP Devices**

This session is divided into two main stages:

1. **Backup Process** – Extraction of the *bootloader*, *partition table*, and *application firmware* from the ESP device.
2. **Recovery Process** – Restoration of the extracted firmware data onto another ESP device.

---

### Part I – Firmware Backup

1. **Verification of Security Configurations**
   It is essential to verify whether the ESP device employs cryptographic protection mechanisms such as *encrypted flash* or *secure boot*. This can be achieved using the following commands:

   ```bash
   ls /dev/tty*               # Identify the USB port in use
   espefuse.py --port /dev/ttyUSB[X] summary   # '[X]' corresponds to the USB port number
   ```

   Indicators of an **unprotected firmware** include:

   * `FLASH_CRYPT_CNT = 0` → Flash encryption is disabled.
   * `ABS_DONE_0 = False / ABS_DONE_1 = False` → Secure Boot is disabled.

2. **Execution of the Backup Script**
   First, download the script `dump_esp32_firmware.sh` and assign execution permissions:

   ```bash
   chmod +x dump_esp32_firmware.sh
   ./dump_esp32_firmware.sh /dev/ttyUSB[X]   # Replace '[X]' with the appropriate port number
   ```

   Upon successful execution, the following binary files are generated:

   * `app.bin`
   * `bootloader.bin`
   * `partition-table.bin`

   If no errors occur, the backup process is considered complete.

---

### Part II – Firmware Recovery on Another ESP

To restore the firmware onto a different ESP device, ensure that the three binary files are located in the current working directory, and execute the following command:

```bash
esptool.py -p /dev/ttyUSB0 write_flash 0x1000 bootloader.bin 0x8000 partition-table.bin 0x10000 app.bin
```

Successful recovery is confirmed when the output contains:

```
Uploading stub...
Running stub...
Stub running...
Leaving...
Hard resetting via RTS pin...
```

---

### Conclusion

This procedure enables both the preservation and replication of ESP firmware in a reliable manner. The backup phase ensures the secure storage of essential components, while the recovery phase allows for the deployment of identical firmware configurations across multiple devices.


