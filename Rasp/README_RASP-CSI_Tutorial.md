Follow the steps at `Getting Started` in [Nexmonster Repository](https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92)

Credits to Matthias Schulz, Daniel Wegemer, Matthias Hollick, Francesco Gringoli, and Jakob Link for their released work mentioned above.

Upon completion of the steps outlined in the Nexmonster repository, the CSI data acquisition can be performed.

In this experiment, we employed the application [iPerf2](https://play.google.com/store/apps/details?id=iperf.project&hl=en) for Android 

The command line used was following:
```
iperf -c 192.168.1.1 -u -b 500M -t 60 -i 1 -l 1400 -p 5500
```
`-c` $\rightarrow$ access as client;  
`<IP>` $\rightarrow$ AP's IP;  
`-u` $\rightarrow$ packets sent as UDP protocol;  
`-b` $\rightarrow$  ;  
`-t` $\rightarrow$ time that communication in seconds;  
`-l` $\rightarrow$  ; e  
`-p` $\rightarrow$ port to listen (the smae setting up in Raspberry Pi).  
