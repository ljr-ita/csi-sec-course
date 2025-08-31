#!/bin/bash

# ------------------------------
# Script para gerar dump do ESP32 (bootloader + partition table + app)
# Autor: Gioliano Braga / ChatGPT
# ------------------------------

# Porta serial do ESP32 bom
PORT=$1

if [ -z "$PORT" ]; then
  echo "Uso: $0 /dev/ttyUSBX"
  exit 1
fi

# Velocidade da serial
BAUD=460800

# Offsets e tamanhos (ajustar se seu build for diferente)
BOOTLOADER_OFFSET=0x1000
BOOTLOADER_SIZE=0x7000

PARTITION_OFFSET=0x8000
PARTITION_SIZE=0x1000

APP_OFFSET=0x10000
APP_SIZE=0x100000   # 1MB, ajuste conforme seu build real

# Nomes dos arquivos de saída
BOOTLOADER_BIN=bootloader.bin
PARTITION_BIN=partition-table.bin
APP_BIN=app.bin

echo "Iniciando dump do ESP32 na porta $PORT ..."
echo "1) Dump do Bootloader..."
esptool.py -p $PORT -b $BAUD read_flash $BOOTLOADER_OFFSET $BOOTLOADER_SIZE $BOOTLOADER_BIN

echo "2) Dump da Partition Table..."
esptool.py -p $PORT -b $BAUD read_flash $PARTITION_OFFSET $PARTITION_SIZE $PARTITION_BIN

echo "3) Dump do App principal..."
esptool.py -p $PORT -b $BAUD read_flash $APP_OFFSET $APP_SIZE $APP_BIN

echo ""
echo "Dump concluído!"
echo "Arquivos gerados:"
echo "  - $BOOTLOADER_BIN"
echo "  - $PARTITION_BIN"
echo "  - $APP_BIN"
echo ""
echo "Pronto para gravar no outro ESP32 com:"
echo "esptool.py -p /dev/ttyUSB_DESTINO write-flash 0x1000 bootloader.bin 0x8000 partition-table.bin 0x10000 app.bin"
