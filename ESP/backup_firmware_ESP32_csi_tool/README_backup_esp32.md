## Welcome to session of backup ESP firmware  

This session is split into two parts:  
  1. First Part – Backing up the bootloader + partition table + app from the ESP;  
  2. Second Part – Recovering firmware data on another ESP.  

     
### Pt. 1 - Backup
  - Check the cryptography on the ESP (encrypted flash or secure boot):    
	\>>> ls /dev/tty* 		                                # check your USBPort  
    \>>> espefuse.py --port /dev/ttyUSB[**X**] summary    # 'X' is the number corresponding to the USB port connected to the ESP  

  - The following indicators show that the firmware is unprotected:  
    \>>> FLASH_CRYPT_CNT = 0 → indica que a criptografia da flash não está ativada.  
    \>>> ABS_DONE_0 = False / ABS_DONE_1 = False → indica que o Secure Boot não está ativo.  

  - Download the file dump_esp32_firmware.sh and make it executable:  
    \>>> chmod +x dump_esp32_firmware.sh                  # change its permission  
    \>>> ./dump_esp32_firmware.sh /devttyUSB[**X**]       # here, 'X' is the number of the ESP that currently has the firmware installed  

    Running the script will generate three files (bootloader + partition table + app):    
      aap.bin  
      bootloader.bin  
      partition-table.bin  

    If the process completes without errors, the backup is successful.  

### Pt. 2 - Recover on another ESP   
  - In same path that contains the three files generated, run:  
    \>>> esptool.py -p /dev/ttyUSB0 write_flash 0x1000 bootloader.bin 0x8000 partition-table.bin 0x10000 app.bin  

    Success is confirmed if the following output appears:  
      Uploading stub...  
      Running stub...  
      Stub running...  
      
      Leaving...  
      Hard resetting via RTS pin...
    

\>>> **Now, enjoying your new toy!**

