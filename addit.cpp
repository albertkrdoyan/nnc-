#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void LoadX(const char* sourcePath, int len, int slen, double** X) {
	ifstream read;
	read.open(sourcePath);
	int i = 0, j = 0;

	if (read.is_open()) {
		char _c = '\0';
		double num[3] = { 0, 0, 0 };
		int _i = 0;

		while (true) {
			_c = read.get();

			if (_c == -1 || _c == ' ') {
				if (num[2] == 0) {
					X[i][j] = num[0] / 255;
					j++;
					if (j == slen) {
						j = 0;
						i++;
					}
				}
				else {
					while (num[0]-- != 0) {
						X[i][j] = num[1] / 255;
						j++;
						if (j == slen) {
							j = 0;
							i++;
						}
					}
				}

				if (_c == -1)
					break;

				num[0] = num[1] = num[2] = 0;
				_i = 0;
			}
			else if (_c == ':') {
				_i = 1;
				num[2] = 1;
			}
			else {
				num[_i] *= 10;
				num[_i] += _c - '0';
			}
		}

		read.close();		
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);
}

void LoadY(const char* sourcePath, int len, int slen, double** Y) {
	ifstream read;
	read.open(sourcePath);
	int i = 0;

	if (read.is_open()) {
		char _c = '\0';

		while ((_c = read.get()) != -1)
			Y[i++][_c - '0'] = 1.0f;

		read.close();
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);
}