# Project To-Do List

- [ ] Testar as formulas do section.py
- [ ] Implementar Timoshenko 3-node
- [ ] Testar funcao no load distribuido
- [ ] Tentar fazer com integracao numerica como uma solucao paralela da matriz de rigidez
- [ ] Euler-bernoulli 3node está com 8 nós somente(central sem rotacao). Tentar com 9 nós (ver se é viavel).
- [ ] Shear Stress Distribution in Cross Section
- [ ] area nao ta 90 graus acima
- [ ] quando cria varios elementos com o mesh a escala do diagrama fica ruim
- [ ] Fazer com que os pontos sejam precisos na hora de dar hover (Saber para um valor exato de xy)
- [ ] Do diagrama de tensao na secao fazer que o elemento seja o completo nao o subdivido por mesh
- [ ] Verificar Shear Coefficients
- [ ] Implementar possibilidade de digitar o G da propriedade, com verificacao de erro(somente dois podem ser digitados ao mesmo tempo (E,G,v))
- [ ] Testar Springs com restricao rotacional tbm
- [ ] Load Distribution escrever bonitinho em Latex do lado no app.py
- [ ] Procurar por codigo duplicado(deixar mais clean)
- [ ] Verificar distribuicao shear diagram esta estranho para femap_test

# Ajuste do front-end
- [ ] Na aba analyses nao diz o total de nos considerando o mesh
- [ ] Tirar +- dos itens que exigem um valor especifico para rodar(posicoes dos nos, E, G, poisson,etc)
- [ ] Verificar unidades no app (retirar todas), e numero de casa depois da virgula deixar usuario escolher
- [ ] Point Loads inicia em Y force arrumar
- [ ] Arrumar Load de arquivo para rodar deve-se deletar o arquivo
- [ ] Arrumar botar valores as vezes tem que clicar no enter para aceitar o valor
- [ ] No grafico do preview da estrutura fazer com que a escala seja fixa

# Ajustes Professor
- [ ] Verificar Timoshenko com os exemplos do livro (Logan e o Reddy) e colocar mais tipos de carga (carga distribuida.etc).
- [ ] Checar função no load distribuido
- [ ] Tentar fazer com integração numérica como uma solução paralela da matriz de rigidez (baixa prioridade)