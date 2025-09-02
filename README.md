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

3. **Conversão dos dados** em [conversor](https://colab.research.google.com/drive/1FRaAT8DRVYhVs-cR9nTWevtEcgdXA9Oj?usp=sharing)

   * Transformação de dados CSI **complexos/binários** para **amplitude e fase**
   * Scripts em **Python + Jupyter Notebooks**

4. **Filtros, Pré-Processamento e Visualização dos Dados** em [filter](https://colab.research.google.com/drive/1IvP7TYWbTOz2F1XwMMLiYumFG7ECS0Bu?usp=sharing)

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
├── notebooks/            #Jupyter Notebooks e Dashboard
│   ├── app_deploy.py            #Dashboard usando o banco de dados 'DB_Gender'
│   └── Preprocessing*    #Filtros, pré-processamento, visualização de dados, treinamento e classificação de modelos
├── DB_Gender             #Banco de dados para rodar o Dashboard
├── datasets              #Banco de dados handpass e spider-sense
└── README.md
```

---

## 📘 Recursos de Apoio

* **ESP32 CSI**:

  * [esp_csi_tool](https://github.com/espressif/esp-csi/blob/master/README.md)
  * [Espressif](https://github.com/espressif/esp-csi/blob/master/examples/esp-radar/console_test/README.md)

* **Raspberry Pi (Nexmon CSI)**:

  * [Nexmon CSI - seemoo-lab (Oficial)](https://github.com/seemoo-lab/nexmon_csi)
  * [Nexmonster](https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92)

---

## 🚀 Como Usar

1. Clone este repositório:

   ```bash
   git clone https://github.com/ljr-ita/csi-sec-course.git
   cd csi-sec-course
   ```

2. Siga os tutoriais de configuração em **esp32/** e **raspberry/** para preparar os dispositivos.

3. Execute o Dashboard no **Google Colab** em [dash_link](https://colab.research.google.com/drive/1SRxBt9UCCeovy88kPLbYjfSSCQKyL_f4?usp=sharing).

4. Explore os exemplos de **visualização e classificação** com ML em [notebook](https://colab.research.google.com/drive/1n7FFGbKWFlyAUlM74drRR6mJsy-prQG1?usp=sharing)

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
@misc{sbsseg2025_minicurso,
  author       = {Autor(es) do minicurso},
  title        = {SBSeg2025 - Minicurso 4: Wi-Fi Sensing e CSI aplicados à Cibersegurança: Fundamentos, Aplicações e Prática},
  year         = {2025},
  howpublished = {\url{https://}},
  note         = {Acessado em: XX xxx 2025}
}
```
