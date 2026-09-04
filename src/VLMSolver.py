import numpy as np
import numba as nb


class VLMSolver:
    """
    Vortex Lattice Method solver
    """
    #--------------------#
    #   Initialization   #
    #--------------------#
    def __init__(self, surfaces, u_inf, boundary, ratio, a_ratio):

        # Flow properties
        self.U        = u_inf      # inflow velocity [m/s]
        self.boundary = boundary   # enable the free surface condition

        # surfaces
        self.surfaces = surfaces   # list of the VLMSurface

        # global parameters
        self.N_tot    = None       # total number of panels 
        self.controls = None       # control points of all the real wings
        self.normals  = None       # normals of all the real wings

        # Linear system
        self.A = None       # influence coefficient matrix
        self.b = None       # right hand side

        # Numerical parameters
        self.rc      = None       # smallest panel dimension (width or chord) - reference length of the cutoff distance
        self.c_ref   = None       # mean chord - reference length of the vortex core radius
        self.core    = None       # vortex core radius [m] = ratio * c_ref
        self.ratio   = ratio      # vortex core radius / mean chord, for the wake roll-up and the induced-velocity loads.
                                  # A Scully core on a physical length regularises the roll-up of the tip vortex sheet;
                                  # 0.05 is a typical value, results are weakly sensitive between 0.02 and 0.1.
        self.a_ratio = a_ratio    # cutoff / minimum panel width for the boundary condition (influence matrix A and RHS b).
                                  # A and b must share the same cutoff, and a_ratio must be < 0.5 so that a control point
                                  # never falls inside the cutoff of its own ring.

        self.time=0

        self._get_parameters()

    #-------------#
    #   Methods   #
    #-------------#
 
    def _get_parameters(self):
        """ 
        get the global parameters from each surface
        """
        ctrl  = []
        norm  = []
        N_tot = 0
        r_min = np.inf
        chords = []
        for surface in self.surfaces :
            # smallest panel dimension of the lattice, width or chord: the cutoff is a fraction of it, so that no
            # control point ever falls inside the cutoff of a neighbouring segment whatever the panel aspect ratio
            for panel in surface.wing_panels["real"] :
                r_surf = min(np.linalg.norm(panel.width), np.linalg.norm(panel.chord))
                if r_surf < r_min :
                    r_min = r_surf
            N_tot += surface.N * surface.M
            for panel in surface.wing_panels["real"] :
                ctrl.append(panel.ctr)     # we compute the normals and control points vector only once
                norm.append(panel.normal)
                chords.append(surface.M*np.linalg.norm(panel.chord))     # local chord of the section the panel belongs to
        self.N_tot    = N_tot
        self.controls = np.array(ctrl)
        self.normals  = np.array(norm)
        self.rc       = r_min
        self.c_ref    = float(np.mean(chords))
        self.core     = self.ratio*self.c_ref

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _induced_velocity(p, c1, c2, gamma, rc, core):
        """
        Return the velocity induced by the vortex segments [c1, c2], of strentgh gamma,
        at the points p

        Input :
        p     -> points where the velocity is computed, shape (N,3)
        c1    -> starting points of the segments, shape (M,3)
        c2    -> ending points of the segments, shape (M,3)
        gamma -> circulation of each segment, shape (M,)
        rc    -> cutoff distance: a segment closer than rc to the point induces nothing (used for the boundary condition)
        core  -> vortex core radius of a Scully profile: the singular velocity is multiplied by h**2/(h**2 + core**2),
                 h being the distance to the segment (used for the wake roll-up and the induced loads); 0 for no core

        Output :
        v     -> velocity induced at each point by each segment, shape (N,3)
        """
        N = p.shape[0]
        M = c1.shape[0]
        v_total = np.zeros((N, 3), dtype=np.float64)

        eps = rc*1e-10
        inv4pi = 1.0 / (4.0 * np.pi)
        d_cut2 = (rc)**2
        core2  = core**2

        # Parallel loop over points
        for i in nb.prange(N):
            px, py, pz = p[i, 0], p[i, 1], p[i, 2]
            vx = 0.0
            vy = 0.0
            vz = 0.0

            for j in range(M):

                rx1 = px - c1[j, 0]
                ry1 = py - c1[j, 1]
                rz1 = pz - c1[j, 2]

                rx2 = px - c2[j, 0]
                ry2 = py - c2[j, 1]
                rz2 = pz - c2[j, 2]

                r0x = c2[j, 0] - c1[j, 0]
                r0y = c2[j, 1] - c1[j, 1]
                r0z = c2[j, 2] - c1[j, 2]
                r0_norm2 = r0x * r0x + r0y * r0y + r0z * r0z

                cx = ry1 * rz2 - rz1 * ry2
                cy = rz1 * rx2 - rx1 * rz2
                cz = rx1 * ry2 - ry1 * rx2

                cross_norm2 = cx * cx + cy * cy + cz * cz

                r1_norm = np.sqrt(rx1 * rx1 + ry1 * ry1 + rz1 * rz1)
                r2_norm = np.sqrt(rx2 * rx2 + ry2 * ry2 + rz2 * rz2)

                if r0_norm2 > 0.0 and r1_norm > eps and r2_norm > eps and cross_norm2 > 0.0:
                    d2 = cross_norm2 / r0_norm2   # perpendicular distance squared from the point to the segment
                    if d2 >= d_cut2:

                        dot_term = (
                            r0x * (rx1 / r1_norm - rx2 / r2_norm)
                            + r0y * (ry1 / r1_norm - ry2 / r2_norm)
                            + r0z * (rz1 / r1_norm - rz2 / r2_norm)
                        )

                        factor = gamma[j] * inv4pi * dot_term / cross_norm2
                        if core2 > 0.0:
                            factor = factor * d2 / (d2 + core2)     # Scully vortex core

                        vx += cx * factor
                        vy += cy * factor
                        vz += cz * factor
                    # else : contribution ignored
            v_total[i, 0] = vx
            v_total[i, 1] = vy
            v_total[i, 2] = vz
        return v_total
    

    def _vectorize (self, grid, gamma, n) :
        """
        Transform the grid corner points into two array of points that define all
        the segments, needed for the _induced_velocity function, and the local circulation
        of each segment

        Input :
        grid  -> 2D grid of the points that defines the segments which induce the velocity
        gamma -> circulation of each vortex ring
        n     -> number of panels in spanwise direction, needed to reshape the grid and gamma in the right way

        Output :
        c1, c2    -> starting and ending points of each segment
        gamma_seg -> local circulation of each segment
        """
        m = round(np.size(grid)/(3*(n+1)))-1                    # number of panels in chordwise direction
        idx = np.arange((m+1)*(n+1)).reshape(m+1, n+1)          # reshape for building the segments points
        c1_j = grid[idx[:, :-1]]                                # spanwise segments
        c2_j = grid[idx[:, 1:]]
        c1_i = grid[idx[:-1, :]]                                # chordwise segments
        c2_i = grid[idx[1:, :]]
        c1 = np.concatenate([c1_j.reshape(-1,3), c1_i.reshape(-1,3)])
        c2 = np.concatenate([c2_j.reshape(-1,3), c2_i.reshape(-1,3)])
        gamma = gamma.reshape(m, n)
        gamma_seg_j =  gamma[1:, :] - gamma[:-1, :]                                     # local circulation at each spanwise segment except the first and last one
        gamma_seg_j_full = np.vstack([gamma[0, :], gamma_seg_j, -gamma[-1, :]])         # all the local circulations of spanwise segments
        gamma_seg_i =  -gamma[:, 1:] + gamma[:, :-1]                                    # local circulation at each chordwise segment except the first and last one        
        gamma_seg_i_full = np.hstack([-gamma[:, [0]], gamma_seg_i, gamma[:, [-1]]])     # all the local circulations of chordwise segments
        gamma_seg = np.concatenate([gamma_seg_j_full.reshape(-1), gamma_seg_i_full.reshape(-1)])    # local circulation of each segment
        return c1, c2, gamma_seg
    
    def _full_vectorize(self, wing, skip_shed=False):
        """
        Build the segments needed for using _induced_velocity from all the surfaces
        Basically do _vectorize at each surface needed and return the same type of results, directly usable for _induced_velocity

        Input :
            wing      -> True if the wing influence needs to be counted (for the local speed for instance) - False if it doesn't (for the RHS...)
            skip_shed -> True to leave out the wake rings shed at the current time step (trailing edge row, tip columns).
                         Their strength is the unknown of the current step and their influence is in the matrix A (implicit
                         Kutta condition, Katz & Plotkin sect. 13.12). Only meaningful with wing=False.

        Output :
            c1, c2    -> starting and ending points of each segment
            gamma_seg -> local circulation of each segment
        """
        surfaces  = self.surfaces
        c1_tot = np.empty((0, 3))
        c2_tot = np.empty((0, 3))
        gamma_tot = np.empty(0)
        empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
        for surface in surfaces:
            m, n = surface.M, surface.N
            for k in surface.wing.keys():
                l_grid, r_grid = surface.wake[k]["left"], surface.wake[k]["right"]
                n_l = round(len(l_grid)/(m+1)-1)         # number of rings in the tip wakes (time columns)
                n_r = round(len(r_grid)/(m+1)-1)
                if wing :                               # taking the wing into consideration or not
                    mid_grid  = np.concatenate([surface.wing[k][:m*(n+1)], surface.wake[k]["middle"]])
                    mid_gamma = np.concatenate([surface.gamma, surface.gamma_wake["middle"]])
                elif skip_shed :                        # drop the newly shed row / column, keep the older rings only
                    mid_grid  = surface.wake[k]["middle"][n+1:]
                    mid_gamma = surface.gamma_wake["middle"]
                    l_grid    = l_grid.reshape(m+1, n_l+1, 3)[:, :-1].reshape(-1, 3);  n_l -= 1
                    r_grid    = r_grid.reshape(m+1, n_r+1, 3)[:, 1:].reshape(-1, 3);   n_r -= 1
                else :
                    mid_grid  = surface.wake[k]["middle"]
                    mid_gamma = surface.gamma_wake["middle"]
                if len(mid_grid) > n+1 :
                    c1, c2, gamma_v = self._vectorize(mid_grid, mid_gamma, n)
                else :
                    c1, c2, gamma_v = empty
                if ( surface.tip_shed in ("left", "both") ) and n_l > 0 :
                    c1_l, c2_l, gamma_vl = self._vectorize(l_grid, surface.gamma_wake["left"], n_l)
                else :
                    c1_l, c2_l, gamma_vl = empty   # in order to let the concatenation work and to be able to modifie the gamma for boundary condition
                if ( surface.tip_shed in ("right", "both") ) and n_r > 0 :
                    c1_r, c2_r, gamma_vr = self._vectorize(r_grid, surface.gamma_wake["right"], n_r)
                else :
                    c1_r, c2_r, gamma_vr = empty
                c1_tot    = np.concatenate([c1_tot, c1, c1_l, c1_r])  
                c2_tot    = np.concatenate([c2_tot, c2, c2_l, c2_r])  
                if k=="real":           # used to put a - before the mirrored gamma if we want a pure symmetric boundary condition (wall), without a minus its a negative image
                    gamma_tot = np.concatenate([gamma_tot, gamma_v, gamma_vl, gamma_vr]) 
                else :
                    if self.boundary=="antisymmetric" :
                        gamma_tot = np.concatenate([gamma_tot, gamma_v, gamma_vl, gamma_vr]) 
                    else :
                        gamma_tot = np.concatenate([gamma_tot, -gamma_v, -gamma_vl, -gamma_vr]) 
        return c1_tot, c2_tot, gamma_tot

    def _ring_influence(self, v, surface):
        """
        Normal velocity induced at every control point by one vortex ring of unit strength,
        including its image when a free-surface / wall condition is active.

        Input :
            v       -> the four ring corner points, ordered as the panel rings
            surface -> the surface the ring belongs to (needed for the image)

        Output :
            a       -> normal velocity at each control point, shape (N_tot,)
        """
        ctrl, norm = self.controls, self.normals
        rc   = self.rc*self.a_ratio
        one  = np.ones(4)
        c1 = np.array([v[0], v[1], v[2], v[3]]).reshape(-1,3)
        c2 = np.array([v[1], v[2], v[3], v[0]]).reshape(-1,3)
        v_ring = self._induced_velocity(ctrl, c1, c2, one, rc, 0.0)
        if self.boundary=="symmetric" or self.boundary=="antisymmetric" :
            vm = surface._symmetry(np.array(v), False, np.array([0,0,1.]), np.array([0,0,0.]), 0)
            c1 = np.array([vm[0], vm[1], vm[2], vm[3]]).reshape(-1,3)
            c2 = np.array([vm[1], vm[2], vm[3], vm[0]]).reshape(-1,3)
            sign = -1.0 if self.boundary=="symmetric" else 1.0          # wall: negative image ; free surface: positive image
            v_ring = v_ring + self._induced_velocity(ctrl, c1, c2, sign*one, rc, 0.0)
        return np.sum(v_ring * norm, axis=1)

    def _build_A (self, dt=None):
        """
        Construct the influence coefficients matrix A by computing the velocity induced at each control point by each vortex ring,
        and projecting it on the normal direction of the panel.

        If dt is given, the wake rings shed during the current time step are included with the strength of the panel they are
        shed from (implicit Kutta condition, Katz & Plotkin sect. 13.12): the trailing-edge row sheds one ring per panel, the tip
        columns shed one ring per panel when tip shedding is enabled. These rings have a fixed geometry (edge, edge convected by
        u_inf*dt) so A stays constant for a constant time step. Including them in A makes the wing tip-edge vorticity and the
        shed tip-wake edge cancel exactly at every step; with an explicit treatment the cancellation lags by one step and the tip
        circulation relaxes at a rate proportional to the tip panel width.
        """
        n    = self.N_tot
        A = np.zeros((n,n))
        j = 0
        for surface in self.surfaces:
            m, nn = surface.M, surface.N
            for i in range(nn*m):
                A[:,j] = self._ring_influence(surface.wing_panels["real"][i].vrt, surface)
                if dt is not None :
                    row, col = divmod(i, nn)
                    shift = dt*self.U
                    if row == m-1 :                                     # trailing-edge panel: newly shed trailing-edge ring
                        e = surface.edges["middle"]
                        A[:,j] += self._ring_influence([e[col], e[col+1], e[col+1]+shift, e[col]+shift], surface)
                    if col == 0 and surface.tip_shed in ("left", "both") :        # left tip panel: newly shed tip ring
                        e = surface.edges["left"]
                        A[:,j] += self._ring_influence([e[row]+shift, e[row], e[row+1], e[row+1]+shift], surface)
                    if col == nn-1 and surface.tip_shed in ("right", "both") :    # right tip panel: newly shed tip ring
                        e = surface.edges["right"]
                        A[:,j] += self._ring_influence([e[row], e[row]+shift, e[row+1]+shift, e[row+1]], surface)
                j += 1
        self.A = A


    def _build_b (self):
        """ 
        Construct the RHS b
        """
        n        = self.N_tot
        u_inf    = self.U
        ctrl     = self.controls
        normal   = self.normals
        u_inf  = np.tile(u_inf, (n,1))   # n would return a 1D array [u_x, u_y, u_z, u_x, u_y, u_z, ...] the tuple parameters is needed to reshape it in a (n, 3) array
        b = np.zeros(n)
        c1, c2, gamma_v = self._full_vectorize(False, skip_shed=True)                     # wake rings of known strength (shed at previous steps)
        if len(gamma_v) == 0 :                                                             # no wake yet (first step)
            v = np.zeros((n,3))
        else :
            v = self._induced_velocity(ctrl, c1, c2, gamma_v, self.rc*self.a_ratio, 0.0)  # velocity induced at each control point by the wake vortex rings, same cutoff as A
        b = -np.sum((v + u_inf) * normal, axis=1)
        self.b = b


    def _kuttas_loads(self):
        """
        Compute the loads by applying the Kutta-Joukowski theorem to each bound and trailing segment of the wing lattice.

        The lift is evaluated with the free-stream velocity only (linearised Kutta-Joukowski theorem, Katz & Plotkin
        eq. 12.25): F = rho * u_inf x (gamma * segment). The velocity induced by the discrete lattice at a segment
        midpoint is of order gamma / (panel width) and does not converge under mesh refinement, so it is not used for
        the lift. It is kept in a separate term, surface.loads_induced, which carries the induced drag.

        Each surface ends up with
            surface.loads         -> total linearised load vector (bound + trailing segments)
            surface.loads_induced -> load vector due to the induced velocity (bound + trailing segments)
            surface.Cl_2d         -> sectional lift coefficient at each spanwise station (bound segments, linearised)
        """
        u        = self.U
        surfaces = self.surfaces
        u_norm   = np.linalg.norm(u)
        for surface in surfaces:
            m, n        = surface.M, surface.N
            wing_panels = surface.wing_panels
            gamma       = surface.gamma
            u_inf    = np.tile(u, (m*n, 1))          # m*n would return a 1D array [u_x, u_y, u_z, u_x, u_y, u_z, ...] the tuple parameters is needed to reshape it in a (m*n, 3) array
            u_inf_t  = np.tile(u, ((n-1)*m, 1))      # n-1 because at both tip the local circulation is none (we assume steady state for the loads computation)
            circ     = gamma - np.concatenate([np.zeros(n), gamma[:n*(m-1)]])                            # circulation at each bound segment
            circ_t   = gamma.reshape(m, n)[:,1:].reshape(-1) - gamma.reshape(m, n)[:,:-1].reshape(-1)    # circulation at each trailing segment
            width    = []         
            points   = []         
            chords   = []         
            points_t = []        
            for k,p in enumerate(wing_panels["real"]):
                width.append(p.vrt[1]-p.vrt[0])             # width of each bound segment
                points.append((p.vrt[0] + p.vrt[1])/2)      # mid bound segment points
                if k%n != 0 :                                   # tips are not taken into account TODO tip shedding
                    chords.append(p.vrt[0] - p.vrt[3])          # length of each trailing segment
                    points_t.append((p.vrt[0] + p.vrt[3])/2)    # mid trailing segment points
            width    = np.array(width)
            points   = np.array(points)
            chords   = np.array(chords)
            points_t = np.array(points_t)
            # linearised Kutta-Joukowski: free-stream velocity only
            F_bound    = np.cross(u_inf,   circ[:,None]*width)                  # loads of the bound segments
            F_trailing = np.cross(u_inf_t, circ_t[:,None]*chords)               # loads of the trailing segments
            # induced-velocity term, kept apart (induced drag)
            c1, c2, gamma_v = self._full_vectorize(True)
            F_bound_i    = np.cross(self._induced_velocity(points,   c1, c2, gamma_v, 0.0, self.core), circ[:,None]*width)
            F_trailing_i = np.cross(self._induced_velocity(points_t, c1, c2, gamma_v, 0.0, self.core), circ_t[:,None]*chords)
            S = np.sum(np.array([p.area for p in surface.wing_panels["real"]]).reshape(m,n), axis=0)
            p = surface.plan
            surface.Cl_2d         = 2*np.sum(F_bound[:,p+1].reshape(m, n), axis=0)/(S*u_norm*u_norm)   # 2D lift coefficient at each spanwise station, sign reversed if plan is 1 because the z axis points downward
            surface.loads         = np.sum(np.concatenate([F_bound,   F_trailing]),   axis=0)
            surface.loads_induced = np.sum(np.concatenate([F_bound_i, F_trailing_i]), axis=0)
        
    def _secondary_computation (self):
        """ 
        Compute the loads using the pressure difference between the two sides of each panel, and summing up all the contributions


        """
        u_inf = self.U
        for surface in self.surfaces :
            m, n    = surface.M, surface.N
            u_inf   = self.U
            gamma   = surface.gamma
            gamma_w = surface.gamma_wake
            panels  = surface.wing_panels["real"]
            ctrl    = np.array([p.ctr for p in panels])
            normals = np.array([p.normal for p in panels])
            u_inf   = np.tile(u_inf, (n*m, 1))
            d_gam_i = np.concatenate([gamma[n:],gamma_w["middle"][:n]]) - np.concatenate([np.zeros_like(gamma[:n]), gamma[:n*(m-1)]])     # delta circulation at each control point in the chordwise direction   
            # delta circulation at each control point in the spanwise direction
            if surface.tip_shed == "both":  
                n_w = round(len(gamma_w["left"])/m)                            # number of panels in chordwise direction for the tip wake                
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma_w["right"].reshape(m,n_w)[:,0][:,None]]).reshape(-1) - np.hstack([gamma_w["left"].reshape(m,n_w)[:,-1][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)  
            elif surface.tip_shed == "right":
                n_w = round(len(gamma_w["right"])/m)                           # number of panels in chordwise direction for the tip wake
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma_w["right"].reshape(m,n_w)[:,0][:,None]]).reshape(-1) - np.hstack([gamma.reshape(m, n)[:,0][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)
            elif surface.tip_shed == "left": 
                n_w = round(len(gamma_w["left"])/m )                           # number of panels in chordwise direction for the tip wake
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma.reshape(m, n)[:,-1][:,None]]).reshape(-1) - np.hstack([gamma_w["left"].reshape(m,n_w)[:,-1][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)
            else :
                d_gam_j = np.hstack([gamma.reshape(m, n)[:,1:],gamma.reshape(m, n)[:,-1][:,None]]).reshape(-1) - np.hstack([gamma.reshape(m, n)[:,0][:,None], gamma.reshape(m, n)[:,:-1]]).reshape(-1)
            taux_i  = np.array([p.chord/(np.linalg.norm(p.chord)**2) for p in panels])          # chordwise unit vector divided by the mean chord length, for each panel
            taux_j  = np.array([p.width/(np.linalg.norm(p.width)**2) for p in panels])          # spanwise unit vector divided by the mean width, for each panel
            S       = np.array([p.area for p in panels])                                        # area of each panel
            c1, c2, gamma_v = self._full_vectorize(False)
            V       = u_inf + self._induced_velocity(ctrl, c1, c2, gamma_v, 0.0, self.core)
            dp      = np.sum(V * (taux_i*d_gam_i[:,None]/2 + taux_j*d_gam_j[:,None]/2), axis=1) 
            dF      = -dp[:,None]*S[:,None]*normals                                  
            F_tot   = np.sum(dF, axis = 0)
            surface.loads = F_tot

    def _time_sim(self, t, dt, distribution):
        """ 
        Do the time stepping simulation with the wake relaxation

        At each step: (1) the existing wake is convected by u_inf*dt, (2) a new row of rings is shed from the trailing edge
        and, if enabled, from the tips, (3) the circulation is solved with the newly shed rings carrying the current
        circulation of the panels they are shed from (implicit Kutta condition, their influence is in A) and the older rings
        in the right hand side, (4) the wake corner points are convected by the induced velocity (roll-up).

        Input :
            t            -> total simulation time
            dt           -> time step length
            distribution -> type of time step distribution
        """
        # Initialization
        u_inf    = self.U
        surfaces = self.surfaces
        free_sur = self.boundary
        boundary = False
        if free_sur=="symmetric" or free_sur=="antisymmetric" :
            boundary = True
        for surface in surfaces:            # wake initialization by getting the edges
            surface.wake["real"]["middle"] = np.copy(surface.edges["middle"])
            surface.wake["real"]["left"]   = np.copy(surface.edges["left"])   
            surface.wake["real"]["right"]  = np.copy(surface.edges["right"])
            surface.gamma_wake = {"middle":np.array([]), "right":np.array([]), "left":np.array([])}
            surface._update_wake([surface.wake["real"]["middle"], surface.wake["real"]["left"], surface.wake["real"]["right"]])
        match distribution :
            case "classic" :
                dta = np.tile(dt, round(t/dt))
            case "cosine" :         # more pannels at the beginning of the simulation, to better capture the starting vortex
                theta = np.linspace(0, np.pi/2, round(t/dt))  
                dta = t*(1-np.cos(theta))
                dta = dta - np.concatenate([np.array([0]), dta[:-1]])
            case _ :
                raise ValueError(f"Unknown time step distribution '{distribution}', expected 'classic' or 'cosine'")
        dt_A  = None                # time step the current A (and its inverse) was built for
        inv_A = None
        for s in range(round(t/dt)):
            if dt_A is None or abs(dta[s]-dt_A) > 1e-12*abs(dt_A) :     # A depends on dt through the newly shed rings
                self._build_A(dta[s])
                inv_A = np.linalg.inv(self.A)
                dt_A  = dta[s]
            for surface in surfaces:
                n, m   = surface.N, surface.M 
                wake   = surface.wake["real"]["middle"]
                l_wake = surface.wake["real"]["left"]
                r_wake = surface.wake["real"]["right"]
                # simulate the wing advancement
                wake   = wake + dta[s]*u_inf
                l_wake = l_wake + dta[s]*u_inf
                r_wake = r_wake + dta[s]*u_inf
                # shed one row of ring
                wake   = np.concatenate([surface.edges["middle"], wake])                       # for each part of the wake we add its shedding edge to the newly convect wake
                l_wake = np.hstack([l_wake.reshape(m+1, s+1, 3), surface.edges["left"].reshape(m+1, 1, 3)]).reshape(-1, 3)          # vertical concatenation
                r_wake = np.hstack([surface.edges["right"].reshape(m+1, 1, 3), r_wake.reshape(m+1, s+1, 3)]).reshape(-1, 3)         
                surface._update_wake([wake, l_wake, r_wake])        # updating the wake for the next b computation
            # right hand side from the older wake rings, solve for the wing and the newly shed rings
            self._build_b()
            Gamma_tot = inv_A @ self.b
            n_gamma = 0                 # give the start of Gamma in Gamma_tot for each surface
            for surface in surfaces:
                n, m   = surface.N, surface.M 
                Gamma = Gamma_tot[n_gamma:n_gamma+n*m]              # Gamma is the circulation of each panel of this surface
                surface.gamma                = Gamma
                surface.gamma_wake["middle"] = np.concatenate([Gamma[-(n):], surface.gamma_wake["middle"]])         # the newly shed wake rings take the circulation of the edge panels at this step
                surface.gamma_wake["left"]   = np.hstack([surface.gamma_wake["left"].reshape(m, s), Gamma.reshape(m, n)[:,0].reshape(m, 1)]).reshape(-1)            # same with vertical concatenation
                surface.gamma_wake["right"]  = np.hstack([Gamma.reshape(m, n)[:,n-1].reshape(m, 1), surface.gamma_wake["right"].reshape(m, s)]).reshape(-1)
                n_gamma += n*m                                      # updating the position in the total circulation
            # simulate the wake rollup
            c1, c2, gamma_v = self._full_vectorize(True)            # getting all the segments (wing + wake) that induce velocity
            for surface in surfaces:
                n, m   = surface.N, surface.M 
                wake   = surface.wake["real"]["middle"]
                l_wake = surface.wake["real"]["left"]
                r_wake = surface.wake["real"]["right"]
                wake   = wake + np.concatenate([np.zeros((n+1,3)), dta[s]*self._induced_velocity(wake[n+1:], c1, c2, gamma_v, 0.0, self.core)])    # each corner point except at the edges are convect by the induced velocity
                l_wake = l_wake + np.hstack([
                    dta[s]*self._induced_velocity(l_wake.reshape(m+1, s+2, 3)[:,:s+1].reshape(-1, 3), c1, c2, gamma_v, 0.0, self.core).reshape(m+1, s+1, 3),
                    np.zeros((m+1,1,3))
                    ]).reshape(-1, 3)                               # reshape in 2D grid to not take into account the tip edge
                r_wake = r_wake + np.hstack([
                    np.zeros((m+1,1,3)), 
                    dta[s]*self._induced_velocity(r_wake.reshape(m+1, s+2, 3)[:,1:].reshape(-1, 3), c1, c2, gamma_v, 0.0, self.core).reshape(m+1, s+1, 3),
                    ]).reshape(-1, 3)
                surface._update_wake([wake, l_wake, r_wake])
                # at the last step we build the wake panels, usefull for plotting
                if s == round(t/dt)-1 :
                    span   = surface.span
                    panels = []
                    panels = panels + surface._paneling(surface.wake["real"]["middle"], surface.wake["real"]["middle"], 1, n, span)
                    if ( surface.tip_shed == "left" ) or ( surface.tip_shed == "both" ) :
                        panels = panels + surface._paneling(surface.wake["real"]["left"], surface.wake["real"]["left"], 1, s+1, span)
                    if ( surface.tip_shed == "right" ) or ( surface.tip_shed == "both" ) :
                        panels = panels + surface._paneling(surface.wake["real"]["right"], surface.wake["real"]["right"], 1, s+1, span)
                    surface.wake_panels["real"] = panels
                    if boundary:
                        panels = []
                        panels = panels + surface._paneling(surface.wake["mirror"]["middle"], surface.wake["mirror"]["middle"], 1, n, span)
                        if ( surface.tip_shed == "left" ) or ( surface.tip_shed == "both" ) :
                            panels = panels + surface._paneling(surface.wake["mirror"]["left"], surface.wake["mirror"]["left"], 1, s+1, span)
                        if ( surface.tip_shed == "right" ) or ( surface.tip_shed == "both" ) :
                            panels = panels + surface._paneling(surface.wake["mirror"]["right"], surface.wake["mirror"]["right"], 1, s+1, span)
                        surface.wake_panels["mirror"] = panels
