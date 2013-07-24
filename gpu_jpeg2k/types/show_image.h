/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
/*
 * @file show_image.h
 *
 * @author Milosz Ciznicki 
 * @date 03-12-2010
 */

#ifndef SHOW_IMAGE_H_
#define SHOW_IMAGE_H_

void show_image(type_tile *tile);
void show_imagef(type_data *data, int width, int height);
void show_imagec(char *data, int width, int height);

#endif /* SHOW_IMAGE_H_ */
