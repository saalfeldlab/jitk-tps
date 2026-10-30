/*-
 * #%L
 * JavaITK thin plate spline.
 * %%
 * Copyright (C) 2014 - 2022 Howard Hughes Medical Institute.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package jitk.spline;

import static org.junit.Assert.assertEquals;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.Test;

public class JacobianTests
{

	double[][] srcPts;

	double[][] tgtPts;

	double trueDet;

	int ndims;

	int N;

	public void genPtListSimple2d()
	{
		ndims = 2;
		srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0 }, 	// x
				{ 0.0, -1.0, 0.0, 1.0 } };	// y

		tgtPts = new double[][] {
				{ -2.0, 0.0, 2.0, 0.0 },	// x
				{ 0.0, -2.0, 0.0, 2.0 } };	// y
	}

	@Test
	public void testJacobianAffine()
	{
		genPtListSimple2d();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );

		final double[] p = new double[ 2 ];
		final double[] q = new double[ 2 ];
		int i = 0;
		for ( double x = -4; x < 4; x++ )
			for ( double y = -4; y < 4; y++ )
			{
				p[ 0 ] = x;
				p[ 1 ] = y;

				tps.apply( p, q );
				final double[][] j = tps.jacobian( p );
				final DMatrixRMaj jacobianMtx = new DMatrixRMaj( j );
				final double jdet = CommonOps_DDRM.det( jacobianMtx );

				assertEquals( "Affine determinant pt (" + i + ")", 4, jdet, 1e-6 );
				assertEquals( 2.0, j[ 0 ][ 0 ], 1e-6 );
				assertEquals( 2.0, j[ 1 ][ 1 ], 1e-6 );

				assertEquals( 0.0, j[ 1 ][ 0 ], 1e-6 );
				assertEquals( 0.0, j[ 0 ][ 1 ], 1e-6 );
				i++;
			}
	}

	/**
	 * The jacobian, J[ i ][ j ] = d f_i / d p_j, must match central finite
	 * differences of apply. Uses non-affine landmarks so the kernel part is non-zero
	 * (a transposed kernel part is invisible for affine transforms).
	 */
	@Test
	public void testJacobianMatchesFiniteDifferences()
	{
		ndims = 2;
		srcPts = new double[][] {
				{ -1.0, 0.0, 1.0, 0.0, 0.5 },
				{ 0.0, -1.0, 0.0, 1.0, 0.5 } };
		tgtPts = new double[][] {
				{ -1.5, 0.0, 1.0, 0.5, 1.5 },
				{ 0.0, -1.0, 0.5, 2.0, 0.0 } };

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );

		final double[] p = { 0.3, -0.4 };
		final double[][] j = tps.jacobian( p );
		final double[][] jFiniteDiff = centralDifferenceJacobian( tps, p, 1e-5 );

		for ( int c = 0; c < ndims; c++ )
			for ( int d = 0; d < ndims; d++ )
				assertEquals( "d f_" + c + " / d p_" + d, jFiniteDiff[ c ][ d ], j[ c ][ d ], 1e-6 );
	}

	/**
	 * Numerically estimates the jacobian of a transform at p using central finite
	 * differences, with the same convention as jacobian: J[ c ][ d ] = d f_c / d p_d.
	 */
	private static double[][] centralDifferenceJacobian( final ThinPlateR2LogRSplineKernelTransform tps, final double[] p, final double h )
	{
		final int n = p.length;
		final double[][] j = new double[ n ][ n ];
		for ( int d = 0; d < n; d++ )
		{
			final double[] pp = p.clone();
			final double[] pm = p.clone();
			pp[ d ] += h;
			pm[ d ] -= h;
			final double[] fp = tps.apply( pp );
			final double[] fm = tps.apply( pm );
			for ( int c = 0; c < n; c++ )
				j[ c ][ d ] = ( fp[ c ] - fm[ c ] ) / ( 2 * h );
		}
		return j;
	}
}
