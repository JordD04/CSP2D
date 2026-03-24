# entry point for CSP2D

import tkinter as tk
from tkinter import messagebox
import random
import numpy as np

from crystal import UnitCell


NUM_ATOMS = 5
FRAME_X = 375
FRAME_Y = 375
FRAME_CENTER = np.asarray([FRAME_X / 2, FRAME_Y / 2])

class CSP2DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSP 2D")

        # Left Panel: The Canvas
        self.canvas = tk.Canvas(root, width=FRAME_X, height=FRAME_Y, bg="white", highlightthickness=1,
                                highlightbackground="gray")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Right Panel: Controls
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(self.control_frame, text="Enter crystal seed:").pack(pady=5)
        self.user_string = tk.Entry(self.control_frame)
        self.user_string.pack(pady=5)

        # Bind the 'Enter' key to CSP
        self.user_string.bind('<Return>', lambda event: self.init_CSP())
        self.drop_button = tk.Button(self.control_frame, text="Start CSP", command=self.init_CSP)
        self.drop_button.pack(pady=10)

    def init_CSP(self):
        self.canvas.delete("all")
        random.seed(self.user_string.get())
        self.crystal = UnitCell(num_atoms=NUM_ATOMS, frame_center=FRAME_CENTER)

        # set up uc in frame
        self.uc_bounds = [[FRAME_CENTER[0] - (self.crystal.lattice_x / 2), FRAME_CENTER[0] + (self.crystal.lattice_x / 2)],
                     [FRAME_CENTER[1] - (self.crystal.lattice_y / 2), FRAME_CENTER[1] + (self.crystal.lattice_y / 2)]]
        self.graphic_cell = self.canvas.create_rectangle(self.uc_bounds[0][0], self.uc_bounds[1][1],
                                     self.uc_bounds[0][1], self.uc_bounds[1][0])
        self.text = self.canvas.create_text(100, 10, text="Energy: ???")

        # set up atoms in canvas
        self.c_atoms = []
        for atom in self.crystal.atoms:
            self.c_atoms.append(self.canvas.create_oval(
                atom.graphic_coords[0], atom.graphic_coords[1],
                atom.graphic_coords[2], atom.graphic_coords[3],
                fill=atom.colour, outline="black"
            ))

        # set up PBC images in canvas
        self.c_images = []
        for image in self.crystal.images:
            self.c_images.append(self.canvas.create_oval(
                image.graphic_coords[0], image.graphic_coords[1],
                image.graphic_coords[2], image.graphic_coords[3],
                fill=image.colour, outline="black"
            ))

        self.take_step(0, 100)


    def take_step(self, num_steps, max_steps):
        if num_steps < max_steps:
            #self.crystal.random_step()
            self.crystal.descend_gradient()
            for atom_ind, atom in enumerate(self.c_atoms):
                graphic_coords = self.crystal.atoms[atom_ind].graphic_coords
                self.canvas.coords(atom, graphic_coords[0], graphic_coords[1], graphic_coords[2], graphic_coords[3])
            for image_ind, image in enumerate(self.c_images):
                graphic_coords = self.crystal.images[image_ind].graphic_coords
                self.canvas.coords(image, graphic_coords[0], graphic_coords[1], graphic_coords[2], graphic_coords[3])

            self.uc_bounds = [
                [FRAME_CENTER[0] - (self.crystal.lattice_x / 2), FRAME_CENTER[0] + (self.crystal.lattice_x / 2)],
                [FRAME_CENTER[1] - (self.crystal.lattice_y / 2), FRAME_CENTER[1] + (self.crystal.lattice_y / 2)]]
            self.canvas.coords(self.graphic_cell, self.uc_bounds[0][0], self.uc_bounds[1][1],
                                     self.uc_bounds[0][1], self.uc_bounds[1][0])

            energy = self.crystal.get_energy()
            self.canvas.itemconfigure(self.text, text="Energy: " + str(round(energy, 5)))

            # Schedule the next frame (16 = 60fps, 32 = 32fps, 64 = 16fps)
            self.root.after(16, self.take_step, num_steps + 1, max_steps)



if __name__ == "__main__":
    root = tk.Tk()
    app = CSP2DApp(root)
    root.mainloop()