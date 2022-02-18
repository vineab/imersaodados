# Jogo de adivinhar o número
import random
print('Olá, qual seu nome?')
name = input()
print('Então, ' + name + ', estou pensando em um numero entre 1 e 20, tente adivinhar!')
secretNumber = random.randint(1,20)

for guessesTaken in range(1,7):
    guess = int(input())
    if guess < secretNumber:
        print('Você chutou um número menor. Tenta de novo.')
    elif guess > secretNumber:
        print('Você chutou um número maior. Tenta de novo.')
    else:
        break # condição caso a tentativa seja correta (nem maior ou menor, igual)

if guess == secretNumber:
    print('Você acertou, ' + name + '!' + ' Você usou ' + str(guessesTaken) + ' tentativas.')
else:
    print('Não, na verdade eu estava pensando no número ' + str(secretNumber) + '...')
