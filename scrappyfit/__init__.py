"""scrappyFIT - light-element PIXE spectrum fitting and mapping.

An independent Python implementation of the physics in CSIRO's GeoPIXE,
extended where the light-element regime needed it. It reads GeoPIXE's own file
formats, so the two can be used side by side and compared directly.

Layout
------
    io        readers for OMDAQ and GeoPIXE files - LMF list mode, .dam
              Dynamic Analysis matrices, .yield layer models, hubbell.dat
    physics   atomic databases, fluorescence yields, mass attenuation
              coefficients, ionisation cross sections, thick-target yields,
              layered samples, secondary fluorescence
    fitting   peak shape, detector response, SNIP background, least squares
    analysis  elemental maps, region masking, database benchmarking
    gui       the desktop application

What differs from GeoPIXE, and why
----------------------------------
Everything here was added because a light-element spectrum exposed a limit of
the original:

  * mass attenuation coefficients are selectable at run time, and a 'mixed'
    rule (FFAST below each absorber's K edge and below 1 keV, XCOM above)
    follows Heirwegh 2014, which measured it as the most accurate choice for
    silicate standards
  * L and M line data are rebuilt from xraylib rather than the shipped table,
    which carried fabricated intensities for Ca, Sc, Ti and V
  * the detector response adds the terms GeoPIXE omits - Si LVV Auger escape
    and contact injection from the window metallisation - without which the
    fit invents elements that are not in the sample
  * fitted L and M assignments are screened against Coster-Kronig and against
    the element's own K lines before being believed
"""

__version__ = '0.3.0'
__all__ = ['io', 'physics', 'fitting', 'analysis']
