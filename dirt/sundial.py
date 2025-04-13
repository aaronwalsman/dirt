import jax
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass

from dirt.constants import DEFAULT_FLOAT_DTYPE

@static_dataclass
class SundialParams:
    t0 : float = 0
    steps_per_day : float = 240.
    days_per_year : float = 360.
    
    planetary_axis : float = 0 #float(jnp.radians(22.5))
    lattitude : float = 0 #float(jnp.radians(45.))

@static_dataclass
class SundialState:
    t : float
    day : int
    day_progress : float
    year : int
    year_progress : float
    
    sun_direction : jnp.ndarray
    clipped_sun_direction : jnp.ndarray

def sundial(params, dtype=DEFAULT_FLOAT_DTYPE):
    
    def _current_day_year(t):
        day = jnp.floor(t / params.steps_per_day)
        day_progress = t/params.steps_per_day - day
        
        year = jnp.floor(t / (params.steps_per_day * params.days_per_year))
        year_progress = t / (params.steps_per_day * params.days_per_year) - year
        
        #jax.debug.print('tcdy {t} {d} {dp} {y} {yp} ({spd}) ({dpy})',
        #    t=t, d=day, dp=day_progress, y=year, yp=year_progress, spd=params.steps_per_day, dpy=params.days_per_year)
        
        return day, day_progress, year, year_progress
    
    def _compute_sun_direction(t, day_progress, year_progress):
        day_angle = day_progress * jnp.pi * 2
        sd = jnp.sin(day_angle)
        cd = jnp.cos(day_angle)
        year_angle = year_progress * jnp.pi * 2
        sy = jnp.sin(year_angle)
        cy = jnp.cos(year_angle)
        sl = jnp.sin(params.lattitude)
        cl = jnp.cos(params.lattitude)
        sa = jnp.sin(params.planetary_axis)
        ca = jnp.cos(params.planetary_axis)
        sun = jnp.array([1.,0.,0.], dtype=dtype)
        r_lat = jnp.array([
            [  1,  0,  0],
            [  0, cl,-sl],
            [  0, sl, cl],
        ])
        r_day = jnp.array([
            [ cd,  0, sd],
            [  0,  1,  0],
            [-sd,  0, cd],
        ])
        r_axis = jnp.array([
            [  1,  0,  0],
            [  0, ca,-sa],
            [  0, sa, ca],
        ])
        r_year = jnp.array([
            [ cy,  0, sy],
            [  0,  1,  0],
            [-sy,  0, cy],
        ])
        
        sun_direction = r_year @ r_axis @ r_day @ r_lat @ sun
        clipped_sun_direction = jnp.where(
            sun_direction[2] < 0.,
            jnp.zeros_like(sun_direction),
            sun_direction,
        )
        
        #jax.debug.print('l {l} pa {pa} da {da} ya {ya} sd {sd}',
        #    l=params.lattitude,
        #    pa=params.planetary_axis,
        #    da=day_angle,
        #    ya=year_angle,
        #    sd=sun_direction,
        #)
        
        return sun_direction, clipped_sun_direction
    
    def init():
        t = params.t0
        day, day_progress, year, year_progress = _current_day_year(t)
        sun_direction, clipped_sun_direction = _compute_sun_direction(
            t, day_progress, year_progress)
        return SundialState(
            t,
            day,
            day_progress,
            year,
            year_progress,
            sun_direction,
            clipped_sun_direction,
        )
    
    def step(state, dt=1.):
        t = state.t + dt
        day, day_progress, year, year_progress = _current_day_year(t)
        sun_direction, clipped_sun_direction = _compute_sun_direction(
            t, day_progress, year_progress)
        return SundialState(
            t,
            day,
            day_progress,
            year,
            year_progress,
            sun_direction,
            clipped_sun_direction,
        )
    
    return init, step
