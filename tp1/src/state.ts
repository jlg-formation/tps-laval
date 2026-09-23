// ---------------------------------------------------------------------------
// État partagé de l'application et données de démonstration.
// ---------------------------------------------------------------------------

/** Un point posé sur le canvas, en coordonnées pixels. */
export interface Point {
  px: number;
  py: number;
  label: 0 | 1; // 0 = blanc, 1 = noir
}

export interface AppState {
  points: Point[];
  activeLabel: 0 | 1; // classe déposée au prochain clic
  learningRate: number;
  speed: number; // nombre de pas d'apprentissage par frame
  training: boolean;
  cursor: number; // index de l'exemple courant dans la boucle d'entraînement
}

export const state: AppState = {
  points: [],
  activeLabel: 1,
  learningRate: 0.1,
  speed: 8,
  training: false,
  cursor: 0,
};

/**
 * Deux amas de points nettement séparables par une droite, pour illustrer
 * la convergence dès le premier lancement.
 */
export function makeDemoPoints(width: number, height: number): Point[] {
  const points: Point[] = [];
  const rnd = (min: number, max: number) => Math.random() * (max - min) + min;

  // Amas « noir » en haut-gauche.
  for (let i = 0; i < 12; i++) {
    points.push({
      px: rnd(0.1, 0.4) * width,
      py: rnd(0.1, 0.45) * height,
      label: 1,
    });
  }

  // Amas « blanc » en bas-droite.
  for (let i = 0; i < 12; i++) {
    points.push({
      px: rnd(0.6, 0.9) * width,
      py: rnd(0.55, 0.9) * height,
      label: 0,
    });
  }

  return points;
}
