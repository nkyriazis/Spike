#pragma once
#include <cassert>
#include <iostream>

#ifndef NDEBUG
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
    (
#ifndef _MSC_VER
        abi::__cxa_demangle(typeid(TR).name(), nullptr,
            nullptr, nullptr),
#else
        nullptr,
#endif
        std::free
    );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

// From http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c#comment63837522_81870 :
//#define TYPEID_NAME(x) abi::__cxa_demangle(typeid((x)).name(), NULL, NULL, NULL)
#define TYPEID_NAME(x) type_name<decltype(x)>()
#endif

#define SPIKE_ADD_BACKEND_GETSET(TYPE, SUPER)           \
  void backend(std::shared_ptr<Backend::TYPE>&& ptr) {  \
    _backend = ptr;                                     \
    SUPER::backend(ptr);                                \
  }                                                     \
  inline Backend::TYPE* backend() const {               \
    assert(_backend != nullptr &&                       \
           "Need to have backend initialized!");        \
    return (Backend::TYPE*)_backend.get();              \
  }                                                     \
  inline void prepare_backend() {                       \
    prepare_backend_early();                            \
    backend()->prepare();                               \
    prepare_backend_late();                             \
  }

#define SPIKE_MAKE_BACKEND_CONSTRUCTOR(TYPE)            \
  TYPE(::TYPE* front, Context* ctx) {                   \
    _frontend = (void*)front;                           \
    context = ctx;                                      \
  }

#define SPIKE_MAKE_INIT_BACKEND(TYPE)                              \
  void TYPE::init_backend(Context* ctx) {                          \
    auto ptr = ::Backend::TYPE::factory[ctx->backend](this, ctx);  \
    backend(std::shared_ptr<::Backend::TYPE>(ptr));                \
    prepare_backend();                                             \
  } /* Below, instantiate backend factory map: */                  \
  namespace Backend { FactoryMap<::TYPE, TYPE> TYPE::factory; }    \

#define SPIKE_MAKE_STUB_INIT_BACKEND(TYPE)                             \
  void TYPE::init_backend(Context* ctx) {                              \
    assert("This type's backend cannot be instantiated!" && false);    \
  }

#define SPIKE_ADD_BACKEND_FACTORY(TYPE)                          \
  static ::Backend::FactoryMap<::TYPE, ::Backend::TYPE> factory; \
  inline ::TYPE* frontend() const {                              \
    assert(_frontend != nullptr &&                               \
           "Need to have backend initialized!");                 \
    return (::TYPE*)_frontend;                                   \
  }

#define SPIKE_EXPORT_BACKEND_TYPE(BACKEND, TYPE)                          \
  namespace Backend {                                                     \
    namespace BACKEND {                                                   \
      namespace Registry {                                                \
        class TYPE {                                                      \
        public:                                                           \
          static ::Backend::TYPE* factory(::TYPE* front, Context* ctx) {  \
            return new ::Backend::BACKEND::TYPE(front, ctx);              \
          }                                                               \
          TYPE() {                                                        \
            ::Backend::TYPE::factory[#BACKEND] = factory;                 \
          }                                                               \
        };                                                                \
        TYPE TYPE ## _registrar;                                          \
      }                                                                   \
    }                                                                     \
  }

#define STRINGIFY(s) #s
