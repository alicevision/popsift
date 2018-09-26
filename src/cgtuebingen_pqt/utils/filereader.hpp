#ifndef FILEREADER_HPP
#define FILEREADER_HPP

#include <stdexcept>
#include <string>

template<typename T>
class FileReader {
  std::ifstream *fin;
  uint n_;
  uint d_;
  std::string filename;
 public:
  FileReader(std::string fs) {
    filename = fs;
    openFile();
    parseHeader();
  }
  ~FileReader() {
    fin->close();
  }

  const uint num() const {
    return n_;
  }

  const uint dim() const {
    return d_;
  }

  T* data() {
    return data(n_, 0);
  }
  T* data(size_t _num, size_t _offset = 0) {
    size_t offset = _offset * d_;
    size_t length = _num * d_;

    T * buf = new T[length];

    uint8_t *raw_data = new uint8_t[length];
    readContent(filename.c_str(), raw_data, length, offset);

    for (int i = 0; i < length; i++)
      buf[i] = raw_data[i];

    delete[] raw_data;

    return buf;
  }
 private:
  void openFile() {
    fin = new std::ifstream(filename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!fin->good()) {
      fin->close();
      throw std::runtime_error("cannot open file " + filename);
    }
  }
  void parseHeader() {
    fin->seekg(0, std::ios::beg);
    (*fin) >> n_;
    (*fin) >> d_;
  }

  void readContent(std::string fs, uint8_t *ptr, size_t len, size_t offset = 0) {
    fin->seekg(0, std::ios::beg);
    fin->seekg(20 + sizeof(uint8_t) * offset, std::ios::beg);
    fin->read((char*) ptr, (len) * sizeof(uint8_t));
    fin->close();

  }
};

template<>
class FileReader<int> {
  std::ifstream *fin;
  uint n_;
  uint d_;
  std::string filename;
 public:
  FileReader<int>(std::string fs) {
    filename = fs;
    openFile();
    parseHeader();
  }
  ~FileReader<int>() {
    fin->close();
  }

  const uint entries() const {
    return n_;
  }

  const uint dimension() const {
    return d_;
  }

  int* data() {
    return data(n_, 0);
  }
  int* data(size_t _num, size_t _offset) {
    size_t offset = _offset * d_;
    size_t length = _num * d_;
    int *buf = new int[length];
    fin->seekg(0, std::ios::beg);
    fin->seekg(20 + sizeof(int) * offset, std::ios::beg);
    fin->read((char*) buf, (length) * sizeof(int));
    fin->close();

    return buf;
  }
 private:
  void openFile() {
    fin = new std::ifstream(filename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!fin->good()) {
      fin->close();
      throw std::runtime_error("cannot open file " + filename);
    }
  }
  void parseHeader() {
    fin->seekg(0, std::ios::beg);
    (*fin) >> n_;
    (*fin) >> d_;
  }

  void readContent(std::string fs, int *ptr, size_t len, size_t offset = 0) {
    fin->seekg(0, std::ios::beg);
    fin->seekg(20 + sizeof(int) * offset, std::ios::beg);
    fin->read((char*) ptr, (len) * sizeof(int));
    fin->close();

  }
};


template<typename T>
void rread(std::string fs, T *ptr, size_t len, size_t offset = 0) {
  std::ifstream fin(fs.c_str(), std::ios_base::in | std::ios_base::binary);

  if (!fin.good()) {
    fin.close();
    throw std::runtime_error("write error");
  }

  size_t n_ = 0;
  size_t d_ = 0;

  fin >> n_;
  fin >> d_;
  fin.ignore();

  cout << "tellg: " << fin.tellg() << endl;
  cout << "offset: " << (sizeof(T) * offset) << " len: " << len << endl;

  fin.seekg(0, std::ios::beg);
  fin.seekg(20 + sizeof(T) * offset, std::ios::beg);
  fin.read((char*) ptr, (len) * sizeof(T));
  fin.close();

}


float* readFloat(const char* _fn, size_t _dim, size_t _num, size_t _offset) {

  size_t offset = _offset * _dim;
  size_t length = _num * _dim;

  float * buf = new float[length];

  uint8_t *raw_data = new uint8_t[length];
  rread<uint8_t>(_fn, raw_data, length, offset);

  for (int i = 0; i < length; i++)
    buf[i] = raw_data[i];

  delete[] raw_data;

  return buf;
}


#endif