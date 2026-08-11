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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Transforms points with a thin plate spline fit from a csv of landmark pairs.
 * <p>
 * Usage: {@code TpsCsvApply landmarks.csv points.csv [ndims]}
 * <p>
 * The landmark csv holds the moving coordinates in the first {@code ndims}
 * columns and the target coordinates in the next {@code ndims}; {@code ndims}
 * defaults to the column count of the points csv. Blank lines, {@code #}
 * comments and a non-numeric first row (a header) are skipped.
 */
public class TpsCsvApply
{
	/** Reads a numeric csv as rows of values (point-major). */
	public static double[][] readCsv( final String path ) throws IOException
	{
		final List< double[] > rows = new ArrayList<>();
		try ( final BufferedReader reader = new BufferedReader( new FileReader( path ) ) )
		{
			String line;
			int lineno = 0;
			while ( ( line = reader.readLine() ) != null )
			{
				lineno++;
				final int comment = line.indexOf( '#' );
				if ( comment >= 0 )
					line = line.substring( 0, comment );
				line = line.trim();
				if ( line.isEmpty() )
					continue;

				final String[] parts = line.split( "," );
				final double[] values = new double[ parts.length ];
				try
				{
					for ( int i = 0; i < parts.length; i++ )
						values[ i ] = Double.parseDouble( parts[ i ].trim() );
				}
				catch ( final NumberFormatException e )
				{
					if ( rows.isEmpty() )
						continue; // only the first data row may be a header
					throw new IOException( path + " line " + lineno + ": " + e.getMessage() );
				}
				if ( !rows.isEmpty() && values.length != rows.get( 0 ).length )
					throw new IOException( path + " line " + lineno + ": has " + values.length
							+ " columns, expected " + rows.get( 0 ).length );
				rows.add( values );
			}
		}
		if ( rows.isEmpty() )
			throw new IOException( path + ": no data rows" );

		return rows.toArray( new double[ 0 ][] );
	}

	/**
	 * Builds a spline from landmark rows, where each row holds a moving point
	 * followed by its target point. Note that jitk wants landmarks
	 * dimension-major, as {@code double[ndims][nLandmarks]}, whereas the csv and
	 * scipy are both point-major.
	 */
	public static ThinPlateR2LogRSplineKernelTransform fit( final double[][] landmarks, final int ndims )
	{
		final double[][] source = new double[ ndims ][ landmarks.length ];
		final double[][] target = new double[ ndims ][ landmarks.length ];
		for ( int i = 0; i < landmarks.length; i++ )
		{
			if ( landmarks[ i ].length < 2 * ndims )
				throw new IllegalArgumentException( "landmark row " + i + " has "
						+ landmarks[ i ].length + " columns, need at least " + 2 * ndims );
			for ( int d = 0; d < ndims; d++ )
			{
				source[ d ][ i ] = landmarks[ i ][ d ];
				target[ d ][ i ] = landmarks[ i ][ ndims + d ];
			}
		}
		return new ThinPlateR2LogRSplineKernelTransform( ndims, source, target );
	}

	public static void main( final String[] args ) throws IOException
	{
		if ( args.length < 2 )
		{
			System.err.println( "usage: TpsCsvApply landmarks.csv points.csv [ndims]" );
			System.exit( 1 );
		}

		final double[][] landmarks = readCsv( args[ 0 ] );
		final double[][] points = readCsv( args[ 1 ] );
		final int ndims = args.length > 2 ? Integer.parseInt( args[ 2 ] ) : points[ 0 ].length;

		if ( points[ 0 ].length != ndims )
		{
			System.err.println( args[ 1 ] + " has " + points[ 0 ].length + " columns, expected " + ndims );
			System.exit( 1 );
		}

		final ThinPlateR2LogRSplineKernelTransform tps = fit( landmarks, ndims );

		final StringBuilder sb = new StringBuilder();
		final double[] result = new double[ ndims ];
		for ( final double[] point : points )
		{
			tps.apply( point, result );
			for ( int d = 0; d < ndims; d++ )
			{
				if ( d > 0 )
					sb.append( ',' );
				sb.append( String.format( "%.17g", result[ d ] ) );
			}
			sb.append( '\n' );
		}
		System.out.print( sb );
	}
}
