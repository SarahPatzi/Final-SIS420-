import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# Inicialización de Pygame
pygame.init()

# Configuración de la pantalla
display_width = 350  # Ancho de la pantalla en píxeles
display_height = 200  # Altura de la pantalla en píxeles
screen = pygame.display.set_mode((display_width, display_height))  # Crear la ventana del juego
pygame.display.set_caption('Cruza la Calle- Q-Learning')  # Título de la ventana
clock = pygame.time.Clock()  # Reloj para controlar la velocidad de actualización

# Carga de imágenes
dogimg = pygame.image.load('images/PERRITOBONITO.png')  # Imagen del perro
backgroundImg = pygame.image.load('images/fondobonito.png')  # Fondo del juego
redcarimg = pygame.image.load('images/autorojo_reescalado.png')  # Imagen del auto rojo
bluecarimg = pygame.image.load('images/autoazul_reescalado.png')  # Imagen del auto azul

# Parámetros del entorno
dog_start_pos = [175, 175]  # Posición inicial del perro
car1_start_pos = [50, 100]  # Posición inicial del auto rojo
car2_start_pos = [250, 50]  # Posición inicial del auto azul
actions = ['SUBIR', 'BAJAR', 'IZQ', 'DER', 'QUIETO']  # Acciones disponibles para el agente
state_space = (display_width // 25, display_height // 25)  # Espacio de estados dividido en una cuadrícula
action_space = len(actions)  # Número de acciones posibles
num_Episodios = 5000  # Número de episodios para entrenar
max_steps = 100  # Máximo de pasos por episodio
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 1.0  # Probabilidad inicial de exploración
epsilon_decay_rate = 0.995  # Tasa de reducción de epsilon por episodio
min_epsilon = 0.1  # Valor mínimo de epsilon

# Inicialización de la tabla Q con ceros
Q_table = np.zeros((*state_space, action_space))

#funcion para resetear el juego
def reset_game():
    """
    Reinicia las posiciones del perro y los autos al comienzo de un nuevo episodio.
    """
    dog_pos = dog_start_pos[:]  # Copia de la posición inicial del perro
    car1_pos = car1_start_pos[:]  # Copia de la posición inicial del auto rojo
    car2_pos = car2_start_pos[:]  # Copia de la posición inicial del auto azul
    return dog_pos, car1_pos, car2_pos


def take_action(action, dog_pos):
    """
    Actualiza la posición del perro en función de la acción seleccionada.
    """
    if action == 'SUBIR' and dog_pos[1] > 0:  # Mover hacia arriba si no está en el borde superior
        dog_pos[1] -= 25
    elif action == 'BAJAR' and dog_pos[1] < display_height - 25:  # Mover hacia abajo si no está en el borde inferior
        dog_pos[1] += 25
    elif action == 'IZQ' and dog_pos[0] > 0:  # Mover hacia la izquierda si no está en el borde izquierdo
        dog_pos[0] -= 25
    elif action == 'DER' and dog_pos[0] < display_width - 25:  # Mover hacia la derecha si no está en el borde derecho
        dog_pos[0] += 25
    return dog_pos


def move_cars(car1_pos, car2_pos):
    """
    Actualiza las posiciones de los autos en cada paso.
    """
    car1_pos[0] += 5  # Mover el auto rojo hacia la derecha
    car2_pos[0] -= 5  # Mover el auto azul hacia la izquierda
    if car1_pos[0] > display_width:  # Si el auto rojo sale por el borde derecho, reaparece por el izquierdo
        car1_pos[0] = -50
    if car2_pos[0] < -50:  # Si el auto azul sale por el borde izquierdo, reaparece por el derecho
        car2_pos[0] = display_width
    return car1_pos, car2_pos


def check_collision(dog_pos, car1_pos, car2_pos):
    """
    Verifica si el perro colisiona con alguno de los autos.
    """
    dog_rect = pygame.Rect(dog_pos[0], dog_pos[1], 25, 25)  # Crear un rectángulo para el perro
    car1_rect = pygame.Rect(car1_pos[0], car1_pos[1], 50, 25)  # Crear un rectángulo para el auto rojo
    car2_rect = pygame.Rect(car2_pos[0], car2_pos[1], 50, 25)  # Crear un rectángulo para el auto azul
    return dog_rect.colliderect(car1_rect) or dog_rect.colliderect(car2_rect)  # Colisión con cualquier auto


def compute_recompensa(dog_pos):
    """
    Calcula la recompensa en función de la posición del perro.
    """
    if dog_pos[1] == 0:  # Si el perro alcanza la meta (parte superior de la pantalla)
        return 100
    return -1  # Penalización por cada paso


def state_to_index(pos):
    """
    Convierte la posición del perro en un índice para acceder a la tabla Q.
    """
    return int(pos[0] // 25), int(pos[1] // 25)


def train(num_Episodios, max_steps):
    """
    Entrena al agente utilizando el algoritmo de Q-learning.
    """
    global epsilon  # Usamos epsilon globalmente para ajustarlo durante el entrenamiento
    recompensas_per_Episodio = []  # Lista para almacenar las recompensas acumuladas por episodio

    for Episodio in range(num_Episodios):
        # Reinicia el entorno para el episodio actual
        dog_pos, car1_pos, car2_pos = reset_game()
        total_recompensa = 0  # Recompensa acumulada en este episodio
        terminated, truncated = False, False  # Indicadores de estado del episodio
        visualize = Episodio % 100 == 0  # Visualizar solo cada 100 episodios

        for step in range(max_steps):
            # Convertir el estado actual en índices para la tabla Q
            state = state_to_index(dog_pos)

            # Política epsilon-greedy
            if random.uniform(0, 1) < epsilon:  # Exploración
                action_idx = random.randint(0, action_space - 1)
            else:  # Explotación
                action_idx = np.argmax(Q_table[state])

            action = actions[action_idx]  # Acción seleccionada
            dog_pos = take_action(action, dog_pos)  # Actualizar la posición del perro
            car1_pos, car2_pos = move_cars(car1_pos, car2_pos)  # Mover los autos

            # Verificar colisión
            if check_collision(dog_pos, car1_pos, car2_pos):
                recompensa = -100  # Penalización fuerte por colisión
                Q_table[state][action_idx] = (1 - alpha) * Q_table[state][action_idx] + alpha * recompensa
                terminated = True
                break

            recompensa = compute_recompensa(dog_pos)  # Calcular la recompensa
            total_recompensa += recompensa

            # Actualización de Q-valor
            new_state = state_to_index(dog_pos)  # Nuevo estado
            max_future_q = np.max(Q_table[new_state])  # Mejor valor futuro posible
            Q_table[state][action_idx] = (1 - alpha) * Q_table[state][action_idx] + alpha * (recompensa + gamma * max_future_q)

            if recompensa == 100:  # Si alcanza la meta
                terminated = True
                break

            if step == max_steps - 1:  # Si alcanza el máximo de pasos
                truncated = True

            # Visualización del episodio
            if visualize:
                screen.blit(backgroundImg, (0, 0))
                screen.blit(dogimg, dog_pos)
                screen.blit(redcarimg, car1_pos)
                screen.blit(bluecarimg, car2_pos)
                pygame.display.update()
                clock.tick(10)

        recompensas_per_Episodio.append(total_recompensa)  # Guardar la recompensa del episodio

        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)

        # Imprimir progreso cada 100 episodios
        if Episodio % 100 == 0:
            print(f'Episodio: {Episodio}, Recompensa: {total_recompensa}, Ganado: {terminated}, Perdido: {truncated}, Epsilon: {epsilon:.4f}')

    return recompensas_per_Episodio


# Entrenamiento
recompensas = train(num_Episodios, max_steps)

# Graficar resultados detallados
average_recompensas = [np.mean(recompensas[i:i + 100]) for i in range(0, len(recompensas), 100)]
plt.figure(figsize=(10, 5))
plt.plot(recompensas, label='Recompensa por Episodio')
plt.plot(range(0, num_Episodios, 100), average_recompensas, label='Promedio cada 100 Episodios')
plt.xlabel('Episodios')
plt.ylabel('Recompensas')
plt.title('Recompensas durante el Entrenamiento')
plt.legend()
plt.show()

# Exportar la tabla Q-learning
np.savetxt("qlearning_table.txt", Q_table.reshape(-1, Q_table.shape[-1]), fmt="%.4f", delimiter=",",
           header="Q-learning Table (State-Action Values)")

print("Entrenamiento completado. Tabla de Q-learning exportada como 'qlearning_table.txt'.")

# Finalizar Pygame
pygame.quit()
