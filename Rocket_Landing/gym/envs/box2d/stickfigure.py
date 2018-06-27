import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

FPS = 50
SCALE = 0.05

VIEWPORT_H = 150
VIEWPORT_W = 800
H = VIEWPORT_H * SCALE
W = VIEWPORT_W * SCALE

FIGURE_SCALE = 1.5

HEAD_RADIUS = 8 * SCALE * FIGURE_SCALE
BODY_WIDTH = 3 * SCALE * FIGURE_SCALE
SEGMENT_LENGTH = 4 * BODY_WIDTH
MAX_MOTOR_TORQUE = 30.0
GROUNDHEIGHT = H / 20

MAX_STEPS = 2000


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        for i in [0, 4]:
            if self.env.objects[5 + i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.objects[5 + i].ground_contact = True

    def EndContact(self, contact):
        for i in [0, 4]:
            if self.env.objects[5 + i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.objects[5 + i].ground_contact = False


class StickFigure(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None
        self.world = Box2D.b2World()

        self.objects = []
        self.joints = []

        high = np.array([1] * 16)
        self.action_space = spaces.Box(-1, +1, (10,))
        self.observation_space = spaces.Box(-high, high)

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.objects:
            return

        self.world.contactListener = None
        for obj in self.objects:
            self.world.DestroyBody(obj)

        self.objects = []
        self.joints = []

    def _reset(self):
        self._destroy()

        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.stepnumber = 0
        ground = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(0, GROUNDHEIGHT - BODY_WIDTH), (W, GROUNDHEIGHT - BODY_WIDTH), (W, GROUNDHEIGHT),
                              (0, GROUNDHEIGHT)]),
                friction=0.8,
                restitution=0.1)
        )
        ground.color1 = rgb(255, 255, 255)
        self.objects.append(ground)

        initial_x = W / 4
        initial_y = GROUNDHEIGHT + 3 * SEGMENT_LENGTH + HEAD_RADIUS * 3

        head = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=HEAD_RADIUS, pos=(0, 0)),
                density=0.25,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.1)
        )
        head.color1 = rgb(255, 255, 255)
        self.objects.append(head)

        upper_body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y - HEAD_RADIUS),
            angle=0.1,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-BODY_WIDTH / 2, 0),
                                             (+BODY_WIDTH / 2, 0),
                                             (BODY_WIDTH / 2, -SEGMENT_LENGTH),
                                             (-BODY_WIDTH / 2, -SEGMENT_LENGTH))), density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.1)
        )
        upper_body.color1 = rgb(255, 255, 255)
        self.objects.append(upper_body)

        neck_joint = revoluteJointDef(
            bodyA=head,
            bodyB=upper_body,
            localAnchorA=(0, -HEAD_RADIUS),
            localAnchorB=(0, 0),
            enableLimit=True,
            lowerAngle=-np.pi / 4,
            upperAngle=np.pi / 4,
            maxMotorTorque=MAX_MOTOR_TORQUE,
            motorSpeed=0.0,
            enableMotor=True
        )
        neck_joint = self.world.CreateJoint(neck_joint)
        self.joints.append(neck_joint)

        lower_body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y - HEAD_RADIUS - BODY_WIDTH * 4),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-BODY_WIDTH / 2, 0),
                                             (+BODY_WIDTH / 2, 0),
                                             (BODY_WIDTH / 2, -SEGMENT_LENGTH),
                                             (-BODY_WIDTH / 2, -SEGMENT_LENGTH))), density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.1)
        )
        lower_body.color1 = rgb(255, 255, 255)
        self.objects.append(lower_body)

        back_joint = revoluteJointDef(
            bodyA=upper_body,
            bodyB=lower_body,
            localAnchorA=(0, -BODY_WIDTH * 4),
            localAnchorB=(0, 0),
            enableLimit=True,
            lowerAngle=-np.pi / 6,
            upperAngle=np.pi / 3,
            maxMotorTorque=MAX_MOTOR_TORQUE,
            motorSpeed=0.0,
            enableMotor=True
        )
        back_joint = self.world.CreateJoint(back_joint)
        self.joints.append(back_joint)

        for side in (-1, 1):
            # legs
            upper_leg = self.world.CreateDynamicBody(
                position=(initial_x, initial_y - HEAD_RADIUS - 2 * SEGMENT_LENGTH),
                angle=0.5 * side,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=((-BODY_WIDTH / 2, 0),
                                                 (+BODY_WIDTH / 2, 0),
                                                 (BODY_WIDTH / 2, -SEGMENT_LENGTH),
                                                 (-BODY_WIDTH / 2, -SEGMENT_LENGTH))), density=1.0,
                    friction=0.5,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    restitution=0.1)
            )
            upper_leg.color1 = rgb(255, 255, 255)
            self.objects.append(upper_leg)

            upper_leg_joint = revoluteJointDef(
                bodyA=lower_body,
                bodyB=upper_leg,
                localAnchorA=(0, -BODY_WIDTH * 4),
                localAnchorB=(0, 0),
                enableLimit=True,
                lowerAngle=-np.pi / 4,
                upperAngle=np.pi / 4,
                maxMotorTorque=MAX_MOTOR_TORQUE,
                motorSpeed=0.0,
                enableMotor=True
            )
            upper_leg_joint = self.world.CreateJoint(upper_leg_joint)
            self.joints.append(upper_leg_joint)

            lower_leg = self.world.CreateDynamicBody(
                position=(initial_x, initial_y - HEAD_RADIUS - 3 * SEGMENT_LENGTH),
                angle=0.5 * side,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=((-BODY_WIDTH / 2, 0),
                                                 (+BODY_WIDTH / 2, 0),
                                                 (BODY_WIDTH / 2, -SEGMENT_LENGTH),
                                                 (-BODY_WIDTH / 2, -SEGMENT_LENGTH))), density=1.0,
                    friction=0.5,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    restitution=0.1)
            )
            lower_leg.color1 = rgb(255, 255, 255)
            lower_leg.ground_contact = False
            self.objects.append(lower_leg)

            lower_leg_joint = revoluteJointDef(
                bodyA=upper_leg,
                bodyB=lower_leg,
                localAnchorA=(0, -BODY_WIDTH * 4),
                localAnchorB=(0, 0),
                enableLimit=True,
                lowerAngle=-np.pi / 2,
                upperAngle=0,
                maxMotorTorque=MAX_MOTOR_TORQUE,
                motorSpeed=0.0,
                enableMotor=True
            )
            lower_leg_joint = self.world.CreateJoint(lower_leg_joint)
            self.joints.append(lower_leg_joint)

            # arms
            upper_arm = self.world.CreateDynamicBody(
                position=(initial_x, initial_y - HEAD_RADIUS),
                angle=0.5 * side,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=((-BODY_WIDTH / 2, 0),
                                                 (+BODY_WIDTH / 2, 0),
                                                 (BODY_WIDTH / 2, -SEGMENT_LENGTH),
                                                 (-BODY_WIDTH / 2, -SEGMENT_LENGTH))), density=1.0,
                    friction=0.5,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    restitution=0.1)
            )
            upper_arm.color1 = rgb(255, 255, 255)
            self.objects.append(upper_arm)

            upper_arm_joint = revoluteJointDef(
                bodyA=upper_body,
                bodyB=upper_arm,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                enableLimit=True,
                lowerAngle=-np.pi / 2,
                upperAngle=np.pi / 2,
                maxMotorTorque=MAX_MOTOR_TORQUE,
                motorSpeed=0.0,
                enableMotor=True
            )
            upper_arm_joint = self.world.CreateJoint(upper_arm_joint)
            self.joints.append(upper_arm_joint)

            lower_arm = self.world.CreateDynamicBody(
                position=(initial_x, initial_y - HEAD_RADIUS - SEGMENT_LENGTH),
                angle=0.5 * side,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=((-BODY_WIDTH / 2, 0),
                                                 (+BODY_WIDTH / 2, 0),
                                                 (BODY_WIDTH / 2, -SEGMENT_LENGTH),
                                                 (-BODY_WIDTH / 2, -SEGMENT_LENGTH))), density=1.0,
                    friction=0.5,
                    categoryBits=0x0010,
                    maskBits=0x001,
                    restitution=0.1)
            )
            lower_arm.color1 = rgb(255, 255, 255)
            self.objects.append(lower_arm)

            lower_arm_joint = revoluteJointDef(
                bodyA=upper_arm,
                bodyB=lower_arm,
                localAnchorA=(0, -BODY_WIDTH * 4),
                localAnchorB=(0, 0),
                enableLimit=True,
                lowerAngle=0,
                upperAngle=np.pi / 2,
                maxMotorTorque=MAX_MOTOR_TORQUE,
                motorSpeed=0.0,
                enableMotor=True
            )
            lower_arm_joint = self.world.CreateJoint(lower_arm_joint)
            self.joints.append(lower_arm_joint)

        return self._step(np.array([0.0] * 10))[0]

    def _step(self, action):

        assert (len(self.joints) == len(action))

        self.world.Step(1 / FPS, 20, 20)

        for joint, speed in zip(self.joints, action):
            joint.motorSpeed = 2 * np.tanh(float(speed))

        state = []
        for joint in self.joints:
            angle_diff = abs(joint.upperLimit - joint.lowerLimit)
            angle_mean = (joint.lowerLimit + joint.upperLimit) / 2
            angle = joint.angle - (angle_mean - angle_diff / 2)
            angle /= angle_diff
            angle = 2 * (angle - 0.5)
            state.append(angle)

        head_pos = self.objects[1].position
        head_vel = self.objects[1].linearVelocity
        state.extend([2 * (head_pos[1] / H - 0.5),
                      head_vel[0], head_vel[1],
                      self.objects[1].angle,
                      float(self.objects[5].ground_contact),
                      float(self.objects[9].ground_contact)])

        reward = 0.1 * head_vel[0]
        reward -= 0.05 * np.sum(np.abs(action)) + 0.1 * abs(self.objects[1].angularVelocity) + 0.1 * abs(head_vel[1])
        done = self.stepnumber > MAX_STEPS

        if head_pos[0] < 0 or head_pos[1] / H < 0.5:
            done = True
            reward = -1.0
        if head_pos[0] > W:
            done = True
            reward = 1.0

        self.stepnumber += 1

        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

            sky = rendering.FilledPolygon(((0, 0), (0, H), (W, H), (W, 0)))
            sky.set_color(*rgb(0, 0, 0))
            self.viewer.add_geom(sky)

        for obj in self.objects:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, color=obj.color1, filled=False, linewidth=4).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def rgb(r, g, b):
    return r / 255, g / 255, b / 255


if __name__ == "__main__":
    env = StickFigure()
    env.reset()
    max_steps = 300
    while max_steps > 0:
        _, _, d, _ = env.step([0] * 10)
        env.render()
        max_steps -= 1
        if d:
            break
