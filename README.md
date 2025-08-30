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

1. **Configuração dos dispositivos**

   * ESP32 com [esp-csi](https://github.com/espressif/esp-csi)
   * Raspberry Pi4 com [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi)

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
├── esp32/                 # Tutoriais e exemplos para ESP32
├── raspberry/             # Tutoriais e exemplos para Raspberry Pi4 (Nexmon CSI)
├── notebooks/             # Jupyter Notebooks e Dashboard
│   ├── 01-conversao.ipynb  # Conversão de dados CSI
│   └── 02-preprocess.ipynb # Filtros, pré-processamento, visualização de dados, treinamento e classificação de modelos
├── DB_Gender               # Banco de dados para rodar o Dashboard
├── datasets/              # Exemplos de dados coletados
└── README.md              # Este documento
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

3. Execute os notebooks em **Google Colab** (sem necessidade de configuração local).

4. Explore os exemplos de **visualização e classificação** com ML.

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

**Felipe Silveira de Almeida** (ITA e Exército Brasileiro),  
**Eduardo Fabrício Gomes Trindade** (ITA e e Exército Brasileiro),  
**Gioliano de Oliveira Braga** (ITA),  
**Ágney Lopes Roth Ferraz** (ITA),  
**Giovani Hoff da Costa** (ITA),  
**Gustavo Cavalcanti Morais** (ITA) e  
**Lourenço Alves Pereira Júnio** (ITA).


Repositório mantido em: [CSI-Sec-Course](https://github.com/ljr-ita/csi-sec-course)

---

💡 *Prepare seu ambiente, siga o pipeline e venha explorar o futuro do Wi-Fi Sensing aplicado à Cibersegurança!*

---

================================
BibTeX
@misc{sbsseg2025_minicurso,
  author       = {Autor(es) do minicurso},
  title        = {SBSeg2025 - Minicurso 4: Wi-Fi Sensing e CSI aplicados à Cibersegurança: Fundamentos, Aplicações e Prática},
  year         = {2025},
  howpublished = {\url{https://000626cf-7296-4b40-ae6b-d1a550c81174.usrfiles.com/ugd/000626_ae6260b44d4945b0afdfdee0793c24ee.pdf}},
  note         = {Acessado em: 30 ago 2025}
}
