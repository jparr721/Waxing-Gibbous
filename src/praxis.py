#! /usr/bin/env python3
#
def beale_f ( x, n ):
  c1 = 1.5
  c2 = 2.25
  c3 = 2.625

  fx1 = c1 - x[0] * ( 1.0 - x[1]      )
  fx2 = c2 - x[0] * ( 1.0 - x[1] ** 2 )
  fx3 = c3 - x[0] * ( 1.0 - x[1] ** 3 )

  value = fx1 ** 2 + fx2 ** 2 + fx3 ** 2

  return value

def beale_test ( ):
  import platform

  import numpy as np

  n = 2

  print ( '' )
  print ( 'BEALE_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Beale function.' )

  t0 = 0.00001
  h0 = 0.25
  prin = 0

  x = np.array ( [ 0.1, 0.1 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( beale_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, beale_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( beale_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'BEALE_TEST:' )
  print ( '  Normal end of execution.' )
  return

def box_f ( x, n ):
  import numpy as np

  value = 0.0

  for i in range ( 1, 11 ):

    c = - i / 10.0

    fx = np.exp ( c * x[0] ) - np.exp ( c * x[1] ) \
      - x[2] * ( np.exp ( c ) - np.exp ( 10.0 * c ) )

    value = value + fx ** 2

  return value

def box_test ( ):
  import platform

  import numpy as np

  n = 3

  print ( '' )
  print ( 'BOX_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Box function.' )

  t0 = 0.00001
  h0 = 20.0
  prin = 0

  x = np.array ( [ 0.0, 10.0, 20.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( box_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, box_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( box_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'BOX_TEST:' )
  print ( '  Normal end of execution.' )
  return

def chebyquad_f ( x, n ):
  import numpy as np

  fvec = np.zeros ( n )

  for j in range ( 0, n ):

    t1 = 1.0;
    t2 = 2.0 * x[j] - 1.0
    t = 2.0 * t2

    for i in range ( 0, n ):
      fvec[i] = fvec[i] + t2
      th = t * t2 - t1
      t1 = t2
      t2 = th

  for i in range ( 0, n ):
    fvec[i] = fvec[i] / n
    if ( ( i % 2 ) == 1 ):
      fvec[i] = fvec[i] + 1.0 / ( i * ( i + 2 ) )
#
#  Compute F.
#
  value = 0.0
  for i in range ( 0, n ):
    value = value + fvec[i] ** 2

  return value

def chebyquad_test ( ):
  import platform

  import numpy as np

  n = 8

  print ( '' )
  print ( 'CHEBYQUAD_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Chebyquad function.' )

  t0 = 0.00001
  h0 = 0.1
  prin = 0

  x = np.zeros ( n )

  for i in range ( 0, n ):
    x[i] = float ( i + 1 ) / float ( n + 1 )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( chebyquad_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, chebyquad_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( chebyquad_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'CHEBYQUAD_TEST:' )
  print ( '  Normal end of execution.' )
  return

def cube_f ( x, n ):
  fx1 = 10.0 * ( x[1] - x[0] ** 3 )
  fx2 = 1.0 - x[0]

  value = fx1 ** 2 + fx2 ** 2

  return value

def cube_test ( ):
  import platform

  import numpy as np

  n = 2

  print ( '' )
  print ( 'CUBE_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Cube function.' )

  t0 = 0.00001
  h0 = 1.0
  prin = 0

  x = np.array ( [ -1.2, -1.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( cube_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, cube_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( cube_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'CUBE_TEST:' )
  print ( '  Normal end of execution.' )
  return

def helix_f ( x, n ):
  import numpy as np

  r = np.linalg.norm ( x )

  if ( 0.0 <= x[0] ):
    theta = 0.5 * np.arctan2 ( x[1], x[0] ) / np.pi
  else:
    theta = 0.5 * ( np.arctan2 ( x[1], x[0] ) + np.pi ) / np.pi

  fx1 = 10.0 * ( x[2] - 10.0 * theta )
  fx2 = 10.0 * ( r - 1.0 )
  fx3 = x[2]

  value = fx1 ** 2 + fx2 ** 2 + fx3 ** 2

  return value

def helix_test ( ):
  import platform

  import numpy as np

  n = 3

  print ( '' )
  print ( 'HELIX_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Fletcher-Powell Helix function.' )

  t0 = 0.00001
  h0 = 1.0
  prin = 0

  x = np.array ( [ -1.0, 0.0, 0.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( helix_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, helix_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( helix_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'HELIX_TEST:' )
  print ( '  Normal end of execution.' )
  return

def hilbert_f ( x, n ):
  value = 0.0

  for i in range ( 0, n ):
    for j in range ( 0, n ):
      value = value + x[i] * x[j] / ( i + j + 1 )

  return value

def hilbert_test ( ):
  import platform

  import numpy as np

  n = 10

  print ( '' )
  print ( 'HILBERT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Hilbert function.' )

  t0 = 0.00001
  h0 = 10.0
  prin = 0

  x = np.ones ( n )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( hilbert_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, hilbert_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( hilbert_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'HILBERT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def powell3d_f ( x, n ):
  import numpy as np

  value = 3.0 - 1.0 / ( 1.0 + ( x[0] - x[1] ) ** 2 ) \
    - np.sin ( 0.5 * np.pi * x[1] * x[2] ) \
    - np.exp ( - ( ( x[0] - 2.0 * x[1] + x[2] ) / x[1] ) ** 2 )

  return value

def powell3d_test ( ):
  import platform

  import numpy as np

  n = 3

  print ( '' )
  print ( 'POWELL3D_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Powell 3D function.' )

  t0 = 0.00001
  h0 = 1.0
  prin = 0

  x = np.array ( [ 0.0, 1.0, 2.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( powell3d_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, powell3d_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( powell3d_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'POWELL3D_TEST:' )
  print ( '  Normal end of execution.' )
  return

def flin ( n, jsearch, l, f, x, nf, v, q0, q1, qd0, qd1, qa, qb, qc ):
  import numpy as np

  t = np.zeros ( n )
#
#  The search is linear.
#
  if ( 0 <= jsearch ):

    t[0:n] = x[0:n] + l * v[0:n,jsearch]
#
#  The search is along a parabolic space curve.
#
  else:

    qa =                 l * ( l - qd1 ) /       ( qd0 + qd1 ) / qd0
    qb = - ( l + qd0 ) *     ( l - qd1 ) / qd1                 / qd0
    qc =   ( l + qd0 ) * l               / qd1 / ( qd0 + qd1 )

    t[0:n] = qa * q0[0:n] + qb * x[0:n] + qc * q1[0:n]
#
#  The function evaluation counter NF is incremented.
#
  nf = nf + 1
#
#  Evaluate the function.
#
  value = f ( t, n )

  return value, nf, qa, qb, qc

def minfit ( n, tol, a ):
  import numpy as np

  kt_max = 30

  e = np.zeros ( n )
  q = np.zeros ( n )
#
#  Householder's reduction to bidiagonal form.
#
  if ( n == 1 ):
    q[0] = a[0,0]
    a[0,0] = 1.0
    return a, q

  g = 0.0
  x = 0.0

  for i in range ( 0, n ):

    e[i] = g
    l = i + 1

    s = 0.0
    for i1 in range ( i, n ):
      s = s + a[i1,i] ** 2

    g = 0.0

    if ( tol <= s ):

      f = a[i,i]

      g = np.sqrt ( s )
      if ( 0.0 <= f ):
        g = - g

      h = f * g - s
      a[i,i] = f - g

      for j in range ( l, n ):

        f = 0.0
        for i1 in range ( i, n ):
          f = f + a[i1,i] * a[i1,j]
        f = f / h

        for i1 in range ( i, n ):
          a[i1,j] = a[i1,j] + f * a[i1,i]

    q[i] = g

    s = 0.0
    for j1 in range ( l, n ):
      s = s + a[i,j1] ** 2

    g = 0.0

    if ( tol <= s ):

      if ( i < n - 1 ):
        f = a[i,i+1]

      g = np.sqrt ( s )
      if ( 0.0 <= f ):
        g = - g

      h = f * g - s

      if ( i < n - 1 ):

        a[i,i+1] = f - g

        for j1 in range ( l, n ):
          e[j1] = a[i,j1] / h

        for j in range ( l, n ):

          s = 0.0
          for j1 in range ( l, n ):
            s = s + a[j,j1] * a[i,j1]

          for j1 in range ( l, n ):
            a[j,j1] = a[j,j1] + s * e[j1]

    y = abs ( q[i] ) + abs ( e[i] )

    x = max ( x, y )
#
#  Accumulation of right-hand transformations.
#
  a[n-1,n-1] = 1.0
  g = e[n-1]
  l = n - 1

  for i in range ( n - 2, -1, -1 ):

    if ( g != 0.0 ):

      h = a[i,i+1] * g

      for i1 in range ( l, n ):
        a[i1,i] = a[i,i1] / h

      for j in range ( l, n ):

        s = 0.0
        for j1 in range ( l, n ):
          s = s + a[i,j1] * a[j1,j]

        for i1 in range ( l, n ):
          a[i1,j] = a[i1,j] + s * a[i1,i]

    for j1 in range ( l, n ):
      a[i,j1] = 0.0

    for i1 in range ( l, n ):
      a[i1,i] = 0.0

    a[i,i] = 1.0

    g = e[i]

    l = i
#
#  Diagonalization of the bidiagonal form.
#
  epsx = r8_epsilon ( ) * x

  for k in range ( n - 1, -1, -1 ):

    kt = 0

    while ( True ):

      kt = kt + 1

      if ( kt_max < kt ):
        e[k] = 0.0
        print ( '' )
        print ( 'MINFIT - Fatal error!' )
        print ( '  The QR algorithm failed to converge.' )
        exit ( 'MINFIT - Fatal error!' )

      skip = False

      for l2 in range ( k, -1, -1 ):

        l = l2

        if ( abs ( e[l] ) <= epsx ):
          skip = True
          break

        if ( 0 < l ):
          if ( abs ( q[l-1] ) <= epsx ):
            break
#
#  Cancellation of E(L) if 1 < L.
#
      if ( not skip ):

        c = 0.0
        s = 1.0

        for i in range ( l, k + 1 ):

          f = s * e[i]
          e[i] = c * e[i]

          if ( abs ( f ) <= epsx ):
            break

          g = q[i]
#
#  q(i) = h = sqrt(g*g + f*f).
#
          h = r8_hypot ( f, g )

          q[i] = h

          if ( h == 0.0 ):
            g = 1.0
            h = 1.0

          c =   g / h
          s = - f / h
#
#  Test for convergence for this index K.
#
      z = q[k]

      if ( l == k ):
        if ( z < 0.0 ):
          q[k] = - z
          for i1 in range ( 0, n ):
            a[i1,k] = - a[i1,k]
        break
#
#  Shift from bottom 2*2 minor.
#
      x = q[l]
      y = q[k-1]
      g = e[k-1]
      h = e[k]
      f = ( ( y - z ) * ( y + z ) + ( g - h ) * ( g + h ) ) / ( 2.0 * h * y )

      g = r8_hypot ( f, 1.0 )

      if ( f < 0.0 ):
        temp = f - g
      else:
        temp = f + g

      f = ( ( x - z ) * ( x + z ) + h * ( y / temp - h ) ) / x
#
#  Next QR transformation.
#
      c = 1.0
      s = 1.0

      for i in range ( l + 1, k + 1 ):

        g = e[i]
        y = q[i]
        h = s * g
        g = g * c

        z = r8_hypot ( f, h )

        e[i-1] = z

        if ( z == 0.0 ):
          f = 1.0
          z = 1.0

        c = f / z
        s = h / z
        f =   x * c + g * s
        g = - x * s + g * c
        h = y * s
        y = y * c

        for j in range ( 0, n ):
          x = a[j,i-1]
          z = a[j,i]
          a[j,i-1] = x * c + z * s
          a[j,i] = - x * s + z * c

        z = r8_hypot ( f, h )

        q[i-1] = z

        if ( z == 0.0 ):
          f = 1.0
          z = 1.0

        c = f / z
        s = h / z
        f =   c * g + s * y
        x = - s * g + c * y

      e[l] = 0.0
      e[k] = f
      q[k] = x

  return a, q

def minfit_test ( ):
  import platform

  import numpy as np

  n = 5

  print ( '' )
  print ( 'MINFIT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  MINFIT computes part of the SVD of a matrix A.' )
  print ( '    SVD: A = U * D * V\'' )
  print ( '  MINFIT is given A, and returns the diagonal D' )
  print ( '  and the orthogonal matrix V.' )

  a = np.zeros ( [ n, n ] )

  for i in range ( 0, n ):
    a[i,i] = 2.0

  for i in range ( 0, n - 1 ):
    a[i,i+1] = -1.0

  for i in range ( 1, n ):
    a[i,i-1] = -1.0

  r8mat_print ( n, n, a, '  The matrix A:' )
#
#  Numpy's EPS function is not easy to find!
#
  eps = r8_epsilon ( )

  tol = np.sqrt ( eps )

  a, d = minfit ( n, tol, a )

  r8mat_print ( n, n, a, '  The vector V:' )

  r8vec_print ( n, d, '  The singular values D:' )
#
#  Because A is positive definite symmetric, the "missing" matrix V = U.
#
  print ( '' )
  print ( '  Because A is positive definite symmetric,' )
  print ( '  we can reconstruct it as A = V * D * V\'' )

  a2 = np.zeros ( [ n, n ] )

  for i in range ( 0, n ):
    for j in range ( 0, n ):
      for k in range ( 0, n ):
        a2[i,j] = a2[i,j] + a[i,k] * d[k] * a[j,k]

  r8mat_print ( n, n, a2, '  The product A2 = V * D * V\'' )
#
#  Terminate.
#
  print ( '' )
  print ( 'MINFIT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def minny ( n, jsearch, nits, d2, x1, f1, fk, f, x, t, h, v, q0, q1, \
  nl, nf, dmin, ldt, fx, qa, qb, qc, qd0, qd1 ):

  import numpy as np

  machep = r8_epsilon ( )
  small = machep ** 2
  m2 = np.sqrt ( machep )
  m4 = np.sqrt ( m2 )
  sf1 = f1
  sx1 = x1
  k = 0
  xm = 0.0
  fm = fx
  f0 = fx
  dz = ( d2 < machep )
#
#  Find the step size.
#
  s = np.linalg.norm ( x )

  if ( dz ):
    temp = dmin
  else:
    temp = d2

  t2 = m4 * np.sqrt ( abs ( fx ) / temp + s * ldt ) + m2 * ldt
  s = m4 * s + t
  if ( dz and s < t2 ):
    t2 = s

  t2 = max ( t2, small )
  t2 = min ( t2, 0.01 * h )

  if ( fk and f1 <= fm ):
    xm = x1
    fm = f1

  if ( ( not fk ) or abs ( x1 ) < t2 ):

    if ( 0.0 <= x1 ):
      temp = 1.0
    else:
      temp = - 1.0

    x1 = temp * t2

    f1, nf, qa, qb, qc = flin ( n, jsearch, x1, f, x, nf, v, \
      q0, q1, qd0, qd1, qa, qb, qc )

  if ( f1 <= fm ):
    xm = x1
    fm = f1
#
#  Evaluate FLIN at another point and estimate the second derivative.
#
  while ( True ):

    if ( dz ):

      if ( f1 <= f0 ):
        x2 = 2.0 * x1
      else:
        x2 = - x1

      f2, nf, qa, qb, qc = flin ( n, jsearch, x2, f, x, nf, v, \
        q0, q1, qd0, qd1, qa, qb, qc )

      if ( f2 <= fm ):
        xm = x2
        fm = f2

      d2 = ( x2 * ( f1 - f0 ) - x1 * ( f2 - f0 ) ) \
        / ( ( x1 * x2 ) * ( x1 - x2 ) )
#
#  Estimate the first derivative at 0.
#
    d1 = ( f1 - f0 ) / x1 - x1 * d2
    dz = True
#
#  Predict the minimum.
#
    if ( d2 <= small ):

      if ( 0.0 <= d1 ):
        x2 = - h
      else:
        x2 = h

    else:

      x2 = ( - 0.5 * d1 ) / d2

    if ( h < abs ( x2 ) ):

      if ( x2 <= 0.0 ):
        x2 = - h
      else:
        x2 = h
#
#  Evaluate F at the predicted minimum.
#
    ok = True

    while ( True ):

      f2, nf, qa, qb, qc = flin ( n, jsearch, x2, f, x, nf, v, \
        q0, q1, qd0, qd1, qa, qb, qc )

      if ( nits <= k or f2 <= f0 ):
        break

      k = k + 1

      if ( f0 < f1 and 0.0 < x1 * x2 ):
        ok = False
        break

      x2 = 0.5 * x2

    if ( ok ):
      break
#
#  Increment the one-dimensional search counter.
#
  nl = nl + 1

  if ( fm < f2 ):
    x2 = xm
  else:
    fm = f2
#
#  Get a new estimate of the second derivative.
#
  if ( small < abs ( x2 * ( x2 - x1 ) ) ):
    d2 = ( x2 * ( f1 - f0 ) - x1 * ( fm - f0 ) ) / ( ( x1 * x2 ) * ( x1 - x2 ) )
  else:
    if ( 0 < k ):
      d2 = 0.0

  d2 = max ( d2, small )

  x1 = x2
  fx = fm

  if ( sf1 < fx ):
    fx = sf1
    x1 = sx1
#
#  Update X for linear but not parabolic search.
#
  if ( 0 <= jsearch ):
    x[0:n] = x[0:n] + x1 * v[0:n,jsearch]

  return d2, x1, f1, x, nl, nf, fx, qa, qb, qc

def praxis ( t0, h0, n, prin, x, f ):
  import numpy as np
#
#  Initialization.
#
  machep = r8_epsilon ( )
  small = machep * machep
  vsmall = small * small
  large = 1.0 / small
  vlarge = 1.0 / vsmall
  m2 = np.sqrt ( machep )
  m4 = np.sqrt ( m2 )
  scbd = 1.0
  illc = False
  ktm = 1

  if ( illc ):
    ldfac = 0.1
  else:
    ldfac = 0.01

  kt = 0
  nl = 0
  nf = 1
  fx = f ( x, n )
  qf1 = fx
  t = small + abs ( t0 )
  t2 = t
  dmin = small
  h = h0
  h = max ( h, 100.0 * t )
  ldt = h
#
#  The initial set of search directions V is the identity matrix.
#
  v = np.zeros ( [ n, n ] )
  for i in range ( 0, n ):
    v[i,i] = 1.0

  d = np.zeros ( n )
  y = np.zeros ( n )
  z = np.zeros ( n )
  qa = 0.0
  qb = 0.0
  qc = 0.0
  qd0 = 0.0
  qd1 = 0.0
  q0 = x.copy ( )
  q1 = x.copy ( )

  if ( 0 < prin ):
    print2 ( n, x, prin, fx, nf, nl )
#
#  The main loop starts here.
#
  while ( True ):

    sf = d[0]
    d[0] = 0.0
#
#  Minimize along the first direction V(*,1).
#
    jsearch = 0
    nits = 2
    d2 = d[0]
    s = 0.0
    value = fx
    fk = False

    d2, s, value, x, nl, nf, fx, qa, qb, qc = minny ( n, jsearch, nits, \
      d2, s, value, fk, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, \
      fx, qa, qb, qc, qd0, qd1 )

    d[0] = d2

    if ( s <= 0.0 ):
      for i1 in range ( 0, n ):
        v[i1,0] = - v[i1,0]

    if ( sf <= 0.9 * d[0] or d[0] <= 0.9 * sf ):
      d[1:n] = 0.0
#
#  The inner loop starts here.
#
    for k in range ( 2, n + 1 ):

      y = x.copy ( )

      sf = fx

      if ( 0 < kt ):
        illc = True

      while ( True ):

        kl = k
        df = 0.0
#
#  A random step follows, to avoid resolution valleys.
#
        if ( illc ):

          for j in range ( 0, n ):
            r = np.random.rand ( 1 )
            s = ( 0.1 * ldt + t2 * 10.0 ** kt ) * ( r - 0.5 )
            z[j] = s
            x[0:n] = x[0:n] + s * v[0:n,j]

          fx = f ( x, n )
          nf = nf + 1
#
#  Minimize along the "non-conjugate" directions V(*,K),...,V(*,N).
#
        for k2 in range ( k, n + 1 ):

          sl = fx

          jsearch = k2 - 1
          nits = 2
          d2 = d[k2-1]
          s = 0.0
          value = fx
          fk = False

          d2, s, value, x, nl, nf, fx, qa, qb, qc = minny ( n, jsearch, nits, \
            d2, s, value, fk, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, \
            fx, qa, qb, qc, qd0, qd1 )

          d[k2-1] = d2

          if ( illc ):
            s = d[k2-1] * ( ( s + z[k2-1] ) ** 2 )
          else:
            s = sl - fx

          if ( df <= s ):
            df = s
            kl = k2
#
#  If there was not much improvement on the first try, set
#  ILLC = true and start the inner loop again.
#
        if ( illc ):
          break

        if ( abs ( 100.0 * machep * fx ) <= df ):
          break

        illc = True

      if ( k == 2 and 1 < prin ):
        r8vec_print ( n, d, '  The second difference array:' )
#
#  Minimize along the "conjugate" directions V(*,1),...,V(*,K-1).
#
      for k2 in range ( 1, k ):

        jsearch = k2 - 1
        nits = 2
        d2 = d[k2-1]
        s = 0.0
        value = fx
        fk = False

        d2, s, value, x, nl, nf, fx, qa, qb, qc = minny ( n, jsearch, nits, \
          d2, s, value, fk, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, \
          fx, qa, qb, qc, qd0, qd1 )

        d[k2-1] = d2

      f1 = fx
      fx = sf

      for i in range ( 0, n ):
        temp = x[i]
        x[i] = y[i]
        y[i] = temp - y[i]

      lds = np.linalg.norm ( y )
#
#  Discard direction V(*,kl).
#
#  If no random step was taken, V(*,KL) is the "non-conjugate"
#  direction along which the greatest improvement was made.
#
      if ( small < lds ):

        for j in range ( kl - 1, k - 1, -1 ):
          v[0:n,j] = v[0:n,j-1]
          d[j] = d[j-1]

        d[k-1] = 0.0

        v[0:n,k-1] = y[0:n] / lds
#
#  Minimize along the new "conjugate" direction V(*,k), which is
#  the normalized vector:  (new x) - (old x).
#
        jsearch = k - 1
        nits = 4
        d2 = d[k-1]
        value = f1
        fk = True

        d2, lds, value, x, nl, nf, fx, qa, qb, qc = minny ( n, jsearch, nits, \
          d2, lds, value, fk, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, \
          fx, qa, qb, qc, qd0, qd1 )

        d[k-1] = d2

        if ( lds <= 0.0 ):
          lds = - lds
          v[0:n,k-1] = - v[0:n,k-1]

      ldt = ldfac * ldt
      ldt = max ( ldt, lds )

      if ( 0 < prin ):
        print2 ( n, x, prin, fx, nf, nl )

      t2 = m2 * np.linalg.norm ( x ) + t
#
#  See whether the length of the step taken since starting the
#  inner loop exceeds half the tolerance.
#
      if ( 0.5 * t2 < ldt ):
        kt = - 1

      kt = kt + 1

      if ( ktm < kt ):

        if ( 0 < prin ):
          r8vec_print ( n, x, '  X:' )

        value = fx

        return value, x
#
#  The inner loop ends here.
#
#  Try quadratic extrapolation in case we are in a curved valley.
#
    x, q0, q1, nl, nf, fx, qf1, qa, qb, qc, qd0, qd1 = quad ( \
      n, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, fx, qf1, qa, qb, qc, qd0, qd1 )

    for j in range ( 0, n ):
      d[j] = 1.0 / np.sqrt ( d[j] )

    dn = max ( d )

    if ( 3 < prin ):
      r8mat_print ( n, n, v, '  The new direction vectors:' )

    for j in range ( 0, n ):
      v[0:n,j] = ( d[j] / dn ) * v[0:n,j]
#
#  Scale the axes to try to reduce the condition number.
#
    if ( 1.0 < scbd ):

      for i in range ( 0, n ):
        s = 0.0
        for j in range ( 0, n ):
          s = s + v[i,j] ** 2
        s = np.sqrt ( s )
        z[i] = max ( m4, s )

      s = min ( z )

      for i in range ( 0, n ):

        sl = s / z[i]
        z[i] = 1.0 / sl

        if ( scbd < z[i] ):
          sl = 1.0 / scbd
          z[i] = scbd

        v[i,0:n] = sl * v[i,0:n]
#
#  Calculate a new set of orthogonal directions before repeating
#  the main loop.
#
#  Transpose V for MINFIT:
#
    v = np.transpose ( v )
#
#  Call MINFIT to find the singular value decomposition of V.
#
#  This gives the principal values and principal directions of the
#  approximating quadratic form without squaring the condition number.
#
    v, d = minfit ( n, vsmall, v )
#
#  Unscale the axes.
#
    if ( 1.0 < scbd ):

      for i in range ( 0, n ):
        v[i,0:n] = z[i] * v[i,0:n]

      for j in range ( 0, n ):

        s = 0.0
        for i1 in range ( 0, n ):
          s = x + v[i1,j] ** 2
        s = sqrt ( s )

        d[j] = s * d[j]
        v[0:n,j] = v[0:n,j] / s

    for i in range ( 0, n ):

      dni = dn * d[i]

      if ( large < dni ):
        d[i] = vsmall
      elif ( dni < small ):
        d[i] = vlarge
      else:
        d[i] = 1.0 / dni ** 2
#
#  Sort the singular values and singular vectors.
#
    d, v = svsort ( n, d, v )
#
#  Determine the smallest eigenvalue.
#
    dmin = max ( d[n-1], small )
#
#  The ratio of the smallest to largest eigenvalue determines whether
#  the system is ill conditioned.
#
    if ( dmin < m2 * d[0] ):
      illc = True
    else:
      illc = False

    if ( 1 < prin ):

      if ( 1.0 < scbd ):
        r8vec_print ( n, z, '  The scale factors:' )

      r8vec_print ( n, d, '  Principal values of the quadratic form:' )

    if ( 3 < prin ):
      r8mat_print ( n, n, v, '  The principal axes:' )
#
#  The main loop ends here.
#
  if ( 0 < prin ):
    r8vec_print ( n, x, '  X:' )

  value = fx

  return value, x

def print2 ( n, x, prin, fx, nf, nl ):
  print ( '' )
  print ( '  Linear searches      %d' % ( nl ) )
  print ( '  Function evaluations %d' % ( nf ) )
  print ( '  The function value FX = %g' % ( fx ) )

  if ( n <= 4 or 2 < prin ):
    r8vec_print ( n, x, '  X:' )

  return

def quad ( n, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, fx, qf1, qa, qb, \
  qc, qd0, qd1 ):

  import numpy as np

  temp = fx
  fx   = qf1
  qf1  = temp

  temp = x
  x = q1
  q1 = temp

  qd1 = np.linalg.norm ( x - q1 )

  l = qd1
  s = 0.0

  if ( qd0 <= 0.0 or qd1 <= 0.0 or nl < 3 * n * n ):

    fx = qf1
    qa = 0.0
    qb = 0.0
    qc = 1.0

  else:

    jsearch = -1
    nits = 2
    value = qf1
    fk = True

    s, l, value, x, nl, nf, fx, qa, qb, qc = minny ( n, jsearch, nits, \
      s, l, value, fk, f, x, t, h, v, q0, q1, nl, nf, dmin, ldt, \
      fx, qa, qb, qc, qd0, qd1 )

    qa =                 l * ( l - qd1 )       / ( qd0 + qd1 ) / qd0
    qb = - ( l + qd0 )     * ( l - qd1 ) / qd1                 / qd0
    qc =   ( l + qd0 ) * l               / qd1 / ( qd0 + qd1 )

  qd0 = qd1

  xnew = np.zeros ( n )
  xnew[0:n] = qa * q0[0:n] + qb * x[0:n] + qc * q1[0:n]

  q0[0:n] = x[0:n]
  x[0:n] = xnew[0:n]

  return x, q0, q1, nl, nf, fx, qf1, qa, qb, qc, qd0, qd1

def svsort ( n, d, v ):
  for j in range ( 0, n - 1 ):

    j3 = j
    for j2 in range ( j + 1, n ):
      if ( d[j3] < d[j2] ):
        j3 = j2

    t     = d[j]
    d[j]  = d[j3]
    d[j3] = t

    for i in range ( 0, n ):
      t       = v[i,j]
      v[i,j]  = v[i,j3]
      v[i,j3] = t

  return d, v

def svsort_test ( ):
  import platform

  import numpy as np

  n = 5

  print ( '' )
  print ( 'SVSORT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  SVSORT sorts a vector D, and the corresponding columns' )
  print ( '  of a matrix V.' )

  d = np.random.rand ( n )

  v = np.zeros ( [ n, n ] )

  for i in range ( 0, n ):
    for j in range ( 0, n ):
      v[i,j] = 10 * ( i + 1 ) + ( j + 1 )

  print ( '' )
  print ( '  First row = entries of D.' )
  print ( '  Corresponding columns of V below.' )
  print ( '' )
  for j in range ( 0, n ):
    print ( '%14.6g' % ( d[j] ) ),
  print ( '' )
  print ( '' )
  for i in range ( 0, n ):
    for j in range ( 0, n ):
      print ( '%14.6g' % ( v[i,j] ) ),
    print ( '' )

  d, v = svsort ( n, d, v )

  print ( '' )
  print ( '  After sorting D and rearranging V:' )
  print ( '' )
  for j in range ( 0, n ):
    print ( '%14.6g' % ( d[j] ) ),
  print ( '' )
  print ( '' )
  for i in range ( 0, n ):
    for j in range ( 0, n ):
      print ( '%14.6g' % ( v[i,j] ) ),
    print ( '' )
#
#  Terminate.
#
  print ( '' )
  print ( 'SVSORT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def praxis_test ( ):
  import platform

  print ( '' )
  print ( 'PRAXIS_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test the PRAXIS library.' )
#
#  Minimization tests.
#
  beale_test ( )
  box_test ( )
  chebyquad_test ( )
  cube_test ( )
  helix_test ( )
  hilbert_test ( )
  powell3d_test ( )
  rosenbrock_test ( )
  singular_test ( )
  tridiagonal_test ( )
  watson_test ( )
  wood_test ( )
#
#  Utility tests.
#
  minfit_test ( )
  svsort_test ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'PRAXIS_TEST:' )
  print ( '  Normal end of execution.' )
  return

def r8_epsilon ( ):
  value = 2.220446049250313E-016

  return value

def r8_epsilon_test ( ):
  import platform

  print ( '' )
  print ( 'R8_EPSILON_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8_EPSILON produces the R8 roundoff unit.' )
  print ( '' )

  r = r8_epsilon ( )
  print ( '  R = R8_EPSILON()         = %e' % ( r ) )

  s = ( 1.0 + r ) - 1.0
  print ( '  ( 1 + R ) - 1            = %e' % ( s ) )

  s = ( 1.0 + ( r / 2.0 ) ) - 1.0
  print ( '  ( 1 + (R/2) ) - 1        = %e' % ( s ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8_EPSILON_TEST' )
  print ( '  Normal end of execution.' )
  return

def r8_hypot ( x, y ):
  import numpy as np

  if ( abs ( x ) < abs ( y ) ):
    a = abs ( y )
    b = abs ( x )
  else:
    a = abs ( x )
    b = abs ( y )
#
#  A contains the larger value.
#
  if ( a == 0.0 ):
    value = 0.0
  else:
    value = a * np.sqrt ( 1.0 + ( b / a ) ** 2 )

  return value

def r8_hypot_test ( ):
  import platform

  import numpy as np

  print ( '' )
  print ( 'R8_HYPOT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8_HYPOT returns an accurate value for sqrt(A^2+B^2).' )
  print ( '' )
  print ( '             A          B          R8_HYPOT      sqrt(A^2+B^2)' )
  print ( '' )

  b = 2.0

  for i in range ( 0, 20 ):
    a = 1.0
    b = b / 2.0
    c = r8_hypot ( a, b )
    d = np.sqrt ( a ** 2 + b ** 2 )

    print ( '  %12g  %12g  %24.16g  %24.16g' % ( a, b, c, d ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8_HYPOT_TEST' )
  print ( '  Normal end of execution.' )
  return

def r8mat_print ( m, n, a, title ):
  r8mat_print_some ( m, n, a, 0, 0, m - 1, n - 1, title )

  return

def r8mat_print_test ( ):
  import platform

  import numpy as np

  print ( '' )
  print ( 'R8MAT_PRINT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8MAT_PRINT prints an R8MAT.' )

  m = 4
  n = 6
  v = np.array ( [ \
    [ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 ],
    [ 21.0, 22.0, 23.0, 24.0, 25.0, 26.0 ],
    [ 31.0, 32.0, 33.0, 34.0, 35.0, 36.0 ],
    [ 41.0, 42.0, 43.0, 44.0, 45.0, 46.0 ] ], dtype = np.float64 )
  r8mat_print ( m, n, v, '  Here is an R8MAT:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_PRINT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def r8mat_print_some ( m, n, a, ilo, jlo, ihi, jhi, title ):
  incx = 5

  print ( '' )
  print ( title )

  if ( m <= 0 or n <= 0 ):
    print ( '' )
    print ( '  (None)' )
    return

  for j2lo in range ( max ( jlo, 0 ), min ( jhi + 1, n ), incx ):

    j2hi = j2lo + incx - 1
    j2hi = min ( j2hi, n )
    j2hi = min ( j2hi, jhi )

    print ( '' )
    print ( '  Col: ', end = '' )

    for j in range ( j2lo, j2hi + 1 ):
      print ( '%7d       ' % ( j ), end = '' )

    print ( '' )
    print ( '  Row' )

    i2lo = max ( ilo, 0 )
    i2hi = min ( ihi, m )

    for i in range ( i2lo, i2hi + 1 ):

      print ( '%7d :' % ( i ), end = '' )

      for j in range ( j2lo, j2hi + 1 ):
        print ( '%12g  ' % ( a[i,j] ), end = '' )

      print ( '' )

  return

def r8mat_print_some_test ( ):
  import platform

  import numpy as np

  print ( '' )
  print ( 'R8MAT_PRINT_SOME_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8MAT_PRINT_SOME prints some of an R8MAT.' )

  m = 4
  n = 6
  v = np.array ( [ \
    [ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 ],
    [ 21.0, 22.0, 23.0, 24.0, 25.0, 26.0 ],
    [ 31.0, 32.0, 33.0, 34.0, 35.0, 36.0 ],
    [ 41.0, 42.0, 43.0, 44.0, 45.0, 46.0 ] ], dtype = np.float64 )
  r8mat_print_some ( m, n, v, 0, 3, 2, 5, '  Here is an R8MAT:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_PRINT_SOME_TEST:' )
  print ( '  Normal end of execution.' )
  return

def r8vec_print ( n, a, title ):
  print ( '' )
  print ( title )
  print ( '' )
  for i in range ( 0, n ):
    print ( '%6d:  %12g' % ( i, a[i] ) )

def r8vec_print_test ( ):
  import platform

  import numpy as np

  print ( '' )
  print ( 'R8VEC_PRINT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8VEC_PRINT prints an R8VEC.' )

  n = 4
  v = np.array ( [ 123.456, 0.000005, -1.0E+06, 3.14159265 ], dtype = np.float64 )
  r8vec_print ( n, v, '  Here is an R8VEC:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8VEC_PRINT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def rosenbrock_f ( x, n ):
  value = 0.0

  for j in range ( 0, n ):
    if ( ( j % 2 ) == 0 ):
      value = value + ( 1.0 - x[j] ) ** 2
    else:
      value = value + 100.0 * ( x[j] - x[j-1] ** 2 ) ** 2

  return value

def rosenbrock_test ( ):
  import platform

  import numpy as np

  n = 2

  print ( '' )
  print ( 'ROSENBROCK_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Rosenbrock function.' )

  t0 = 0.00001
  h0 = 1.0
  prin = 0

  x = np.array ( [ -1.2, 1.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( rosenbrock_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, rosenbrock_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( rosenbrock_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'ROSENBROCK_TEST:' )
  print ( '  Normal end of execution.' )
  return

def singular_f ( x, n ):
  value = 0.0

  for j in range ( 0, n, 4 ):

    if ( j + 1 <= n - 1 ):
      xjp1 = x[j+1]
    else:
      xjp1 = 0.0

    if ( j + 2 <= n - 1 ):
      xjp2 = x[j+2]
    else:
      xjp2 = 0.0

    if ( j + 3 <= n - 1 ):
      xjp3 = x[j+3]
    else:
      xjp3 = 0.0

    f1 = x[j] + 10.0 * xjp1

    if ( j + 1 <= n - 1 ):
      f2 = xjp2 - xjp3
    else:
      f2 = 0.0

    if ( j + 2 <= n - 1 ):
      f3 = xjp1 - 2.0 * xjp2
    else:
      f3 = 0.0

    if ( j + 3 <= n - 1 ):
      f4 = x[j] - xjp3
    else:
      f4 = 0.0

    value = value \
      +        f1 ** 2 \
      +  5.0 * f2 ** 2 \
      +        f3 ** 4 \
      + 10.0 * f4 ** 4

  return value

def singular_test ( ):
  import platform

  import numpy as np

  n = 4

  print ( '' )
  print ( 'SINGULAR_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Powell Singular function.' )

  t0 = 0.00001
  h0 = 1.0
  prin = 0

  x = np.array ( [ 3.0, -1.0, 0.0, 1.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( singular_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, singular_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( singular_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'SINGULAR_TEST:' )
  print ( '  Normal end of execution.' )
  return

def timestamp ( ):
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

def timestamp_test ( ):
  import platform

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
  print ( '' )

  timestamp ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Normal end of execution.' )
  return

def tridiagonal_f ( x, n ):
  value = x[0] ** 2 + 2.0 * sum ( x[1:n] ** 2 )

  for i in range ( 0, n - 1 ):
    value = value - 2.0 * x[i] * x[i+1]

  value = value - 2.0 * x[0]

  return value

def tridiagonal_test ( ):
  import platform

  import numpy as np

  n = 4

  print ( '' )
  print ( 'TRIDIAGONAL_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Tridiagonal function.' )

  t0 = 0.00001
  h0 = 8.0
  prin = 0

  x = np.zeros ( n )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( tridiagonal_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, tridiagonal_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( tridiagonal_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'TRIDIAGONAL_TEST:' )
  print ( '  Normal end of execution.' )
  return

def watson_f ( x, n ):
  value = 0.0

  for i in range ( 0, 29 ):

    s1 = 0.0
    d = 1.0
    for j in range ( 1, n ):
      s1 = s1 + j * d * x[j]
      d = d * ( i + 1 ) / 29.0

    s2 = 0.0
    d = 1.0
    for j in range ( 0, n ):
      s2 = s2 + d * x[j]
      d = d * ( i + 1 ) / 29.0

    value = value + ( s1 - s2 * s2 - 1.0 ) ** 2

  value = value + x[0] ** 2 + ( x[1] - x[0] ** 2 - 1.0 ) ** 2

  return value

def watson_test ( ):
  import platform

  import numpy as np

  n = 6

  print ( '' )
  print ( 'WATSON_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Watson function.' )

  t0 = 0.00001
  h0 = 1.0
  prin = 0

  x = np.zeros ( n )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( watson_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, watson_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( watson_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'WATSON_TEST:' )
  print ( '  Normal end of execution.' )
  return

def wood_f ( x, n ):
  f1 = x[1] - x[0] ** 2
  f2 = 1.0 - x[0]
  f3 = x[3] - x[2] ** 2
  f4 = 1.0 - x[2]
  f5 = x[1] + x[3] - 2.0
  f6 = x[1] - x[3]

  value = \
      100.0 * f1 ** 2 \
    +         f2 ** 2 \
    +  90.0 * f3 ** 2 \
    +         f4 ** 2 \
    +  10.0 * f5 ** 2 \
    +   0.1 * f6 ** 2

  return value

def wood_test ( ):
  import platform

  import numpy as np

  n = 4

  print ( '' )
  print ( 'WOOD_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  The Wood function.' )

  t0 = 0.00001
  h0 = 10.0
  prin = 0

  x = np.array ( [ -3.0, -1.0, -3.0, -1.0 ] )

  r8vec_print ( n, x, '  Initial point:' )

  print ( '  Function value = %g' % ( wood_f ( x, n ) ) )

  pr, x = praxis ( t0, h0, n, prin, x, wood_f )

  r8vec_print ( n, x, '  Computed minimizer:' )

  print ( '  Function value = %g' % ( wood_f ( x, n ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'WOOD_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  timestamp ( )
  praxis_test ( )
  timestamp ( )

