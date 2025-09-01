Follow the steps at `Getting Started` in [Nexmonster Repository](https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92)

Credits to Matthias Schulz, Daniel Wegemer, Matthias Hollick, Francesco Gringoli, and Jakob Link for their released work mentioned above.

Upon completion of the steps outlined in the Nexmonster repository, the CSI data acquisition can be performed.

In this experiment, we employed the application [iPerf2](https://play.google.com/store/apps/details?id=iperf.project&hl=en) for Android 

The command line used was following:
```
iperf -c 192.168.1.1 -u -b 500M -t 60 -i 1 -l 1400 -p 5500
```
`-c` $\rightarrow$ Access as client;  
`<IP>` $\rightarrow$ AP's IP;  
`-u` $\rightarrow$ Packets sent as UDP protocol;  
`-b` $\rightarrow$  ;  
`-t` $\rightarrow$ Time that communication in seconds;  
`-l` $\rightarrow$  ; e  
`-p` $\rightarrow$ Port to listen (the same setting up in Raspberry Pi). 

Let's to capture CSI data running a script (that [script](https://github.com/ljr-ita/csi-sec-course/blob/main/Rasp/capture_csi.sh) must to be inside your Raspberry):

```
sudo ./capture_csi.sh 3 20 /home/pi/csi.pcap 36/40 15
```
`sudo ./capture_csi.sh` $\rightarrow$ Run the script;  
`3` $\rightarrow$ Time waiting to start capture;  
`20` $\rightarrow$ Capture about to 20 seconds;  
`/home/pi/csi_captured.pcap` $\rightarrow$ Path for outpu file;  
`36/40` $\rightarrow$ Channel/Bandwith (that values must to be the same setting up in [script](https://github.com/ljr-ita/csi-sec-course/blob/main/Rasp/capture_csi.sh))  
`15` $\rightarrow$ Power of TX in dBm  

To access the output file created after to capture you can create a SFTP server:
  1. Connect Raspberry in your networking by cable ou wireless;  
  2. Check the Raspberry IP with `ifconfig`;
  3. Create a SFTP link `sftp <$USER>@<Raspberry_IP>`;
  4. Check all files in current folder (mus be the path when you captured CSI data) `ls -F | grep -v /`;
  5. Get the file `get <file_name.pcap>`.

