<h1 align="center">KenAI</h1>
<p align="center"><i>Repositório criado para compor nota no segundo checkpoint da matéria AI Engeneering Cognitive And Semantic Computation & IOT.</i></p>

## *Sobre o projeto*
Uma inteligência artificial dedicada ao monitoramento de idosos, capaz de discernir entre seres humanos e outros objetos, rastreando suas atividades diárias, como comer, sentar, andar, deitar e levantar. Além disso, é apta a identificar a posição do indivíduo, seja deitado ou sentado, e alertar em caso de movimentos bruscos, como quedas. Também possui a capacidade de reconhecimento facial e identificação de objetos específicos, como remédios, celular e chaves, contribuindo para a segurança e o bem-estar dos idosos.

- Nossa IA será capaz de identificar:
  - um ser humano,
  - movimentação (comer, sentar, andar, deitar, levantar, etc.),
  - posição(deitada, sentada),
  - movimentos bruscos(caídas),
  - reconhecimento facial,
  - objetos especificos(remédios, celular, chaves e etc).

### *Tecnologias*
<p display="inline-block">
  <img width="48" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="python-logo"/>
  <img width="48" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/markdown/markdown-original.svg" alt="markdown-logo"/>
  <img width="48" src="https://assets-global.website-files.com/5f6bc60e665f54545a1e52a5/60e7a074afbe8f09b4e86de5_roboflow_logomark_flat_round.svg" alt="roboflow-logo"/>
</p>

### Bibliotecas usadas
- [OpenCV](https://opencv.org/)
- [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/examples?hl=pt-br)
- [Time](https://docs.python.org/3/library/time.html)
- [Pyttsx3](https://pyttsx3.readthedocs.io/en/latest/)
- [Threading](https://docs.python.org/3/library/threading.html)

### *Rubricas*

<table>
    <thead>
        <th>Questão</th>
        <th>Descrição</th>
        <th>Responsável</th>
    </thead>
    <tbody>
        <tr>
            <td>1.0</td>
            <td>Identificar pessoas</td>
            <td>Alice</td>
        </tr>
        <tr>
            <td>2.0</td>
            <td>Identificar objetos</td>
            <td>Ester</td>
        </tr>
        <tr>
            <td>3.0</td>
            <td>Identificar posição</td>
            <td>Guilherme</td>
        </tr>
        <tr>
            <td>4.0</td>
            <td>Reconhecimento facial </td>
            <td>Larissa</td>
        </tr>
        <tr>
            <td>5.0</td>
            <td>Identificar movimentos bruscos</td>
            <td>Alice</td>
        </tr>
        <tr>
            <td>6.0</td>
            <td>Unir posição com objetos e pessoa para definição de ações</td>
            <td>Mateus</td>
        </tr>
    </tbody>
</table>

### *Instalação*
Antes de executar o código rode este comando para instalar as bibliotecas necessárias:

```
pip install -r requirements.txt
```

### *Execução*
Para executar o código rode este comando:

```
python deteccao_queda.py
```