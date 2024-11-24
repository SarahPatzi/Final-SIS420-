import pygame
import numpy as np

pygame.init()

# Configuración de la pantalla
display_width = 350
display_height = 200
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Cruzar la Calle - Q-Learning Agent')
clock = pygame.time.Clock()

# Imágenes
dogimg = pygame.image.load('images/PERRITOBONITO.png')
backgroundImg = pygame.image.load('images/fondobonito.png')
redimg = pygame.image.load('images/autorojo_reescalado.png')
blueimg = pygame.image.load('images/autoazul_reescalado.png')

# Parámetros del entorno
dog_start_pos = [167.5, 175]  # Inicio cerca de la parte inferior
car1_start_pos = [50, 125]  # Carril 1
car2_start_pos = [250, 75]  # Carril 2
actions = ['SUBIR', 'BAJAR', 'IZQ', 'DER', 'QUIETO']
state_space = (display_width // 25, display_height // 25)  # Dividimos en una cuadrícula de 25x25 píxeles

# Cargar la tabla Q-learning desde archivo
Q_table = np.loadtxt("qlearning_table.txt", delimiter=",")
Q_table = Q_table.reshape((*state_space, len(actions)))


def reset_game():
    """Reinicia las posiciones de el perro y los autos."""
    dos_pos = dog_start_pos[:]
    car1_pos = car1_start_pos[:]
    car2_pos = car2_start_pos[:]
    return dos_pos, car1_pos, car2_pos


def take_action(action, dos_pos):
    """Mueve el perro según la acción tomada."""
    if action == 'SUBIR' and dos_pos[1] > 0:
        dos_pos[1] -= 25
    elif action == 'BAJAR' and dos_pos[1] < display_height - 25:
        dos_pos[1] += 25
    elif action == 'IZQ' and dos_pos[0] > 0:
        dos_pos[0] -= 25
    elif action == 'DER' and dos_pos[0] < display_width - 25:
        dos_pos[0] += 25
    return dos_pos


def move_cars(car1_pos, car2_pos):
    """Actualiza la posición de los autos en cada paso."""
    car1_pos[0] += 5
    car2_pos[0] -= 5
    if car1_pos[0] > display_width:
        car1_pos[0] = -50
    if car2_pos[0] < -50:
        car2_pos[0] = display_width
    return car1_pos, car2_pos


def check_collision(dos_pos, car1_pos, car2_pos):
    """Verifica si el perro colisionó con algún auto."""
    dog_rect = pygame.Rect(dos_pos[0], dos_pos[1], 25, 25)
    car1_rect = pygame.Rect(car1_pos[0], car1_pos[1], 50, 25)
    car2_rect = pygame.Rect(car2_pos[0], car2_pos[1], 50, 25)
    return dog_rect.colliderect(car1_rect) or dog_rect.colliderect(car2_rect)


def state_to_index(pos):
    """Convierte la posición de el perro a un índice para la tabla Q."""
    return int(pos[0] // 25), int(pos[1] // 25)


# Simulación usando la tabla Q-learning
Episodios = 10  # Número de partidas simuladas
for Episodio in range(Episodios):
    dos_pos, car1_pos, car2_pos = reset_game()
    steps = 0
    won = False

    while steps < 100:  # Máximo de pasos por partida
        state = state_to_index(dos_pos)
        action_idx = np.argmax(Q_table[state])  # Elegir la mejor acción según la tabla Q
        action = actions[action_idx]

        # Ejecutar la acción
        dos_pos = take_action(action, dos_pos)
        car1_pos, car2_pos = move_cars(car1_pos, car2_pos)

        # Verificar colisión
        if check_collision(dos_pos, car1_pos, car2_pos):
            print(f"Episodio {Episodio + 1}: el perro perdió tras {steps + 1} pasos.")
            break

        # Verificar si ganó
        if dos_pos[1] == 0:
            print(f"Episodio {Episodio + 1}: el perro ganó en {steps + 1} pasos.")
            won = True
            break

        # Visualización
        screen.blit(backgroundImg, (0, 0))
        screen.blit(dogimg, dos_pos)
        screen.blit(redimg, car1_pos)
        screen.blit(blueimg, car2_pos)
        pygame.display.update()
        clock.tick(10)

        steps += 1

    if not won:
        print(f"Episodio {Episodio + 1}: perro no logró cruzar en 100 pasos.")

pygame.quit()
