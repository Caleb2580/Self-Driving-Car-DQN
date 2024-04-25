import random
import math
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, inp, hid, out, lr):
        
        super().__init__()

        self.inp = inp
        self.hid = hid
        self.out = out

        self.layer_stack = nn.Sequential(
            nn.Linear(inp, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        return self.layer_stack(x)
    


class Car:
    def __init__(self, w, h, x=None):
        self.W = w
        self.H = h

        self.car_width = 70
        self.car_height = 100
        
        if x is None:
            self.x = w/2
            self.y = h - h/4
        else:
            self.x = x
            self.y = self.car_height/2

    def get_rect(self):
        return [self.x-self.car_width/2, self.y - self.car_height/2, self.car_width, self.car_height]

    def move(self):
        self.y += 1
        if self.y - self.car_height/2 > self.H:
            return False
        return True

    def collision(self, cars):
        collisions = 0
        if self.x - self.car_width/2 < 0 or self.x + self.car_width/2 > self.W:
            collisions += 10
        for car in cars:
            if abs(car.y - self.y) < self.car_height and abs(car.x - self.x) < self.car_width:
                collisions += 1
        return collisions

    # def lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
    #     def ccw(A, B, C):
    #         return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    #     def intersect(A, B, C, D):
    #         return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    #     return intersect((x1, y1), (x2, y2), (x3, y3), (x4, y4))

    def line_intersect_distance(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # Check if two lines intersect and return the intersection point and distance
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        def line_length(x1, y1, x2, y2):
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if intersect((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
            # Calculate the intersection point
            A = (x1, y1)
            B = (x2, y2)
            C = (x3, y3)
            D = (x4, y4)
            a1 = y2 - y1
            b1 = x1 - x2
            c1 = a1 * x1 + b1 * y1
            a2 = y4 - y3
            b2 = x3 - x4
            c2 = a2 * x3 + b2 * y3
            determinant = a1 * b2 - a2 * b1

            if determinant == 0:
                return None, None  # Lines are parallel
            else:
                x = (b2 * c1 - b1 * c2) / determinant
                y = (a1 * c2 - a2 * c1) / determinant

                # Calculate the distance from (x1, y1) to the intersection point
                dist = line_length(x1, y1, x, y)
                return (x, y), dist

        return None, None  # No intersection

    def get_detectors(self, rsa, rays, rl, cars):  # rsa - ray spread angle | rays - number of rays | rl - ray length
        detectors = [1 for i in range(rays)]
        angles = [math.pi*.75 + rsa*(i/(rays-1)) for i in range(rays)]
        for i, angle in enumerate(angles):
            x = self.x
            y = self.y  # - self.car_height/2
            e_x = x + rl * math.cos(angle)
            e_y = y + rl * math.sin(angle)

            for car in cars:
                rect = car.get_rect()
                rect2 = [
                    [(rect[0], rect[1]), (rect[0]+rect[2], rect[1])],
                    [(rect[0]+rect[2], rect[1]), (rect[0]+rect[2], rect[1]+rect[3])],
                    [(rect[0]+rect[2], rect[1]+rect[3]), (rect[0], rect[1]+rect[3])],
                    [(rect[0], rect[1]), (rect[0], rect[1]+rect[3])],
                    [(0, 0), (0, self.H)],
                    [(self.W, 0), (self.W, self.H)]
                ]

                for r in rect2:
                    d = self.line_intersect_distance(x, y, e_x, e_y, r[0][0], r[0][1], r[1][0], r[1][1])[1]
                    if d is not None:
                        detectors[i] = min(d/rl, detectors[i])
        
        return detectors

class Environment:
    def __init__(self):
        self.W = 350
        self.H = 600

        self.delayed_reward = 10

        self.n_lanes = 3
        self.lanes = [self.W/(self.n_lanes+1)*i for i in range(1, self.n_lanes+1)]

        self.rays = 15
        self.ray_length = 400
        self.ray_spread_angle = math.pi * 1.5  # 60 degrees spread angle

        self.action_space = [0, 1, 2]
        self.n_actions = len(self.action_space)
        self.n_states = self.rays

        self.reset()
    
    def reset(self):
        self.car = Car(self.W, self.H)

        self.cars = []
        self.interval = self.car.car_height * 3
        self.counter = 0

        self.spawn()

        return self.get_state()

    def spawn(self):
        lane = random.choice(self.lanes)
        self.cars.append(Car(self.W, self.H, lane))

    def step(self, action):
        for i in range(self.delayed_reward):
            if action == 0:
                self.car.x -= 1
            elif action == 2:
                self.car.x += 1

            self.counter += 1

            for car in self.cars:
                if not car.move():
                    self.cars.remove(car)

            if self.counter % self.interval == 0:
                self.spawn()
        
        done = self.car.collision(self.cars)

        reward = 1 - done*2

        return self.get_state(), reward, False

    def get_state(self):
        return self.car.get_detectors(self.ray_spread_angle, self.rays, self.ray_length, self.cars)


if __name__ == '__main__':
    import pygame as pg

    device = 'cuda'

    env = Environment()
    num_states = env.n_states
    num_actions = env.n_actions
    action_space = env.action_space

    model = DQN(num_states, 256, num_actions, .01).to(device)
    model.load_state_dict(torch.load('dqn.pt'))

    pg.init()
    screen = pg.display.set_mode((env.W, env.H))

    running = True

    state = torch.tensor(env.reset()).type(dtype=torch.float).to(device)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                running = False
                break

        screen.fill((255, 255, 255))

        for car in env.cars:
            pg.draw.rect(screen, (0, 0, 255), car.get_rect())

        # state_, reward, done = env.step(random.choice(env.action_space))
        with torch.inference_mode():
            actions = model(torch.tensor(state).type(dtype=torch.float).to(device))
            # print(actions)
            action = actions.argmax().item()
            state, reward, done = env.step(action)

            # Rays
            x = env.car.x
            y = env.car.y  # - env.car.car_height/2
            angles = [math.pi * .75 + env.ray_spread_angle*(i/(env.rays-1)) for i in range(env.rays)]
            for i, angle in enumerate(angles):
                end_x = x + env.ray_length * math.cos(angle)
                end_y = y + env.ray_length * math.sin(angle)
                # color = ((255, 100, 0) if state[i] else (100, 255, 0))
                color = (255, int(255 * state[i]), int(255 * state[i]))
                pg.draw.line(screen, color, (x, y), (end_x, end_y), width=10)

            # Draw car
            pg.draw.rect(screen, (255, 0, 0), env.car.get_rect())

        pg.display.flip()

        pg.time.delay(10)








