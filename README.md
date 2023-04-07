Documentation for setup procedure – in case we need to restart

* Download OS image from https://www.mediafire.com/file/bf7wwnvb6vv9tgb/opencv-4.5.3.56.zip/file
* Install OS image on SD card using Raspberry Pi installer
	Settings:
		- Hostname: testpi.local
		- Enable SSH
		- user: pi
		- password: deepsee
		- Wireless LAN config
* ssh into testpi.local
* sudo raspi-config
	- set max resolution
	- enable VNC
	- enable camera
	- expand filesystem
	- enable predictable network interface names
	- reboot
* sudo apt update
* sudo apt install wireshark
* sudo apt install git
* git clone https://github.com/andrewtratz/deep-see-driving