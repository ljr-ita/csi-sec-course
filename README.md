# 📡 SBSeg2025 - Minicurso 4

## Wi-Fi Sensing e CSI aplicados à Cibersegurança

### Fundamentos, Aplicações e Prática

Bem-vindo ao repositório oficial do **Minicurso 4 do SBSeg 2025**:
👉 *Wi-Fi Sensing e CSI aplicados à Cibersegurança: Fundamentos, Aplicações e Prática*

Este repositório reúne **tutoriais, notebooks e exemplos práticos** para guiar participantes no processo de **configuração de dispositivos, coleta de dados CSI e classificação usando Machine Learning**.

🔗 Link oficial do curso: [CSI-Sec-Course](https://github.com/ljr-ita/csi-sec-course)

---

## 🎯 Objetivo do Minicurso

O minicurso tem como objetivo apresentar **o potencial do Wi-Fi Sensing** aliado à extração de **Channel State Information (CSI)** para aplicações em **cibersegurança**.
Ao final, os participantes terão compreendido:

* Como configurar **ESP32** e **Raspberry Pi 4 (BCM43455c0)** para coletar dados CSI;
* O fluxo completo de processamento de sinais para **extração de amplitude e fase**;
* Técnicas de **filtragem, pré-processamento e visualização de dados**;
* Aplicação de **algoritmos de Machine Learning** para classificação e detecção baseada em CSI.

---

## 🛠️ Pipeline de Aprendizado

O repositório organiza todo o conteúdo em forma de **Pipeline e Dashboard prático**, cobrindo desde a configuração de hardware até os experimentos em ML:

1. **Configuração dos dispositivos** → Vide [recursos](#-recursos-de-apoio)

   * ESP32 com [esp-csi](https://github.com/espressif/esp-csi)
   * Raspberry Pi4 com [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi) e [Nexmonster_CSI](https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92)

2. **Coleta de dados CSI**

   * Utilização do `esp_csi_tool.py` (ESP32)
   * Utilização do `nexmon_csi` (Raspberry Pi4 - **BCM43455c0**)

3. **Conversão dos dados** 

   * Transformação de dados CSI **complexos/binários** para **amplitude e fase**
   * Scripts em **Python + Jupyter Notebooks**

4. **Filtros, Pré-Processamento e Visualização dos Dados**

   * Remoção de **outliers**
   * Normalização e preparação para algoritmos de ML
   * Gráficos e dashboards para inspecionar o CSI processado

6. **Machine Learning aplicado**

   * Demonstrações de algoritmos supervisionados
   * Exemplos de classificação e análise de resultados

---

## 📂 Estrutura do Repositório

```bash
csi-sec-course/
│
├── ESP/                  #Tutoriais e firmware para ESP32 (esp_csi_tool)
│   ├── Conversor_*.ipynb #Conversão de dados CSI em amplitude e fase (para ESP) 
│   ├── filter_esp.ipynb  #Filtro com gráficos para ESP (eliminação de outliers)
│   └── Preprocessing*    #Filtros, pré-processamento, visualização de dados, treinamento e classificação de modelos
├── Rasp/                 #Tutoriais e exemplos para Raspberry Pi4 (Nexmon CSI)
│   ├── Conversor_*.ipynb #Script de captura de dados CSI e conversão em amplitude e fase (para Rasp)
│   ├── filter_rasp.ipynb #Filtro com gráficos para Rasp (eliminação de outliers)
│   └── Preprocessing*    #Filtros, pré-processamento, visualização de dados, treinamento e classificação de modelos
│
└── README.md
```
* **Notebook Spider-sense**: [GoogleColab_Spider-sense](https://colab.research.google.com/drive/1ch9P5nZ40O2V4S31SGjaNqjx1tDylLqc?usp=sharing)
* **Notebook Handpass**: [GoogleColab_Handpass](https://colab.research.google.com/drive/1Ifu2PIgSPsxw4DMxt86Liam9CgiFcEv2?usp=sharing)


---

## 📘 Recursos de Apoio

* **ESP32 CSI**:

  * [esp_csi_tool](https://github.com/espressif/esp-csi/blob/master/README.md)
  * [Espressif](https://github.com/espressif/esp-csi/blob/master/examples/esp-radar/console_test/README.md)

* **Raspberry Pi (Nexmon CSI)**:

  * [Nexmon CSI - seemoo-lab (Oficial)](https://github.com/seemoo-lab/nexmon_csi)
  * [Nexmonster](https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92)

---

## 👥 Público-Alvo

Este minicurso é voltado para:

* Pesquisadores e estudantes interessados em **cibersegurança**, **IoT** e **redes wireless**;
* Entusiastas de **Wi-Fi Sensing e CSI**;
* Profissionais que desejam aprender **como extrair informações de CSI** para aplicações reais.

---

## 📅 SBSeg 2025 - 01 a 04 de setembro de 2025

Este minicurso faz parte do **Simpósio Brasileiro de Cibersegurança (SBSeg 2025)**, um dos maiores eventos da área no Brasil.

---

## ✨ Créditos

Autores:

**Felipe Silveira de Almeida** (ITA e Exército Brasileiro),  `felipefsa@ita.br`  
**Eduardo Fabrício Gomes Trindade** (ITA e Exército Brasileiro),  `trindade@ita.br`  
**Gioliano de Oliveira Braga** (ITA),  `giolianobraga@ita.br`  
**Ágney Lopes Roth Ferraz** (ITA),  `roth@ita.br`  
**Giovani Hoff da Costa** (ITA),  
**Gustavo Cavalcanti Morais** (ITA) e  
**Lourenço Alves Pereira Júnio** (ITA).  `ljr@ita.br`  


Repositório mantido em: [CSI-Sec-Course](https://github.com/ljr-ita/csi-sec-course)

---

💡 *Prepare seu ambiente, siga o pipeline e venha explorar o futuro do Wi-Fi Sensing aplicado à Cibersegurança!*

---

================================
```bibtex
@incollection{Almeida2025_WiFiCSI,
  author    = {Felipe Silveira de Almeida and Eduardo Fabrício Gomes Trindade
               and Gioliano de Oliveira Braga and {\'A}gney Lopes Roth Ferraz
               and Giovani Hoff da Costa and Gustavo Cavalcanti Morais
               and Lourenço Alves Pereira J{\'u}nior},
  title     = {Wi-Fi Sensing e CSI aplicados {\`a} Ciberseguran{\c{c}}a: Fundamentos, Aplica{\c{c}}{\~o}es e Pr{\'a}tica},
  booktitle = {Minicursos do XXV Simp{\'o}sio Brasileiro de Ciberseguran{\c{c}}a},
  editor    = {Diogo Menezes Ferrazani Mattos and C{\'i}ntia Borges Margi
               and Rodrigo Brand{\~a}o Mansilha and Altair Santin
               and Andr{\'e} Gr{\'e}gio and Eduardo Kugler Viegas},
  publisher = {Sociedade Brasileira de Computa{\c{c}}{\~a}o},
  address   = {Porto Alegre, RS, Brasil},
  year      = {2025},
  chapter   = {4},
  pages     = {144--187},
  doi       = {10.5753/sbc.17851.3.4}
}
```
