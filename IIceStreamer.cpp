// **********************************************************************
//
// Copyright (c) 2003-2011 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.4.2
//
// <auto-generated>
//
// Generated from file `IIceStreamer.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#include <IIceStreamer.h>
#include <Ice/LocalException.h>
#include <Ice/ObjectFactory.h>
#include <Ice/BasicStream.h>
#include <IceUtil/Iterator.h>
#include <IceUtil/DisableWarnings.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 304
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 2
#       error Ice patch level mismatch!
#   endif
#endif

static const ::std::string __Streamer__BitmapProvider__setCamera_name = "setCamera";

static const ::std::string __Streamer__BitmapProvider__getBitmap_name = "getBitmap";

::Ice::Object* IceInternal::upCast(::Streamer::BitmapProvider* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::Streamer::BitmapProvider* p) { return p; }

void
Streamer::__read(::IceInternal::BasicStream* __is, ::Streamer::BitmapProviderPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::Streamer::BitmapProvider;
        v->__copyFrom(proxy);
    }
}

void
IceProxy::Streamer::BitmapProvider::setCamera(::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::Streamer::BitmapProvider* __del = dynamic_cast< ::IceDelegate::Streamer::BitmapProvider*>(__delBase.get());
            __del->setCamera(ex, ey, ez, dx, dy, dz, ax, ay, az, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, true, __cnt);
        }
    }
}

::Ice::AsyncResultPtr
IceProxy::Streamer::BitmapProvider::begin_setCamera(::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az, const ::Ice::Context* __ctx, const ::IceInternal::CallbackBasePtr& __del, const ::Ice::LocalObjectPtr& __cookie)
{
    ::IceInternal::OutgoingAsyncPtr __result = new ::IceInternal::OutgoingAsync(this, __Streamer__BitmapProvider__setCamera_name, __del, __cookie);
    try
    {
        __result->__prepare(__Streamer__BitmapProvider__setCamera_name, ::Ice::Normal, __ctx);
        ::IceInternal::BasicStream* __os = __result->__getOs();
        __os->write(ex);
        __os->write(ey);
        __os->write(ez);
        __os->write(dx);
        __os->write(dy);
        __os->write(dz);
        __os->write(ax);
        __os->write(ay);
        __os->write(az);
        __os->endWriteEncaps();
        __result->__send(true);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __result->__exceptionAsync(__ex);
    }
    return __result;
}

void
IceProxy::Streamer::BitmapProvider::end_setCamera(const ::Ice::AsyncResultPtr& __result)
{
    __end(__result, __Streamer__BitmapProvider__setCamera_name);
}

::Streamer::bytes
IceProxy::Streamer::BitmapProvider::getBitmap(::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__Streamer__BitmapProvider__getBitmap_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::Streamer::BitmapProvider* __del = dynamic_cast< ::IceDelegate::Streamer::BitmapProvider*>(__delBase.get());
            return __del->getBitmap(timer, depthOfField, transparentColor, __ctx);
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, true, __cnt);
        }
    }
}

::Ice::AsyncResultPtr
IceProxy::Streamer::BitmapProvider::begin_getBitmap(::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, const ::Ice::Context* __ctx, const ::IceInternal::CallbackBasePtr& __del, const ::Ice::LocalObjectPtr& __cookie)
{
    __checkAsyncTwowayOnly(__Streamer__BitmapProvider__getBitmap_name);
    ::IceInternal::OutgoingAsyncPtr __result = new ::IceInternal::OutgoingAsync(this, __Streamer__BitmapProvider__getBitmap_name, __del, __cookie);
    try
    {
        __result->__prepare(__Streamer__BitmapProvider__getBitmap_name, ::Ice::Normal, __ctx);
        ::IceInternal::BasicStream* __os = __result->__getOs();
        __os->write(timer);
        __os->write(depthOfField);
        __os->write(transparentColor);
        __os->endWriteEncaps();
        __result->__send(true);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __result->__exceptionAsync(__ex);
    }
    return __result;
}

::Streamer::bytes
IceProxy::Streamer::BitmapProvider::end_getBitmap(const ::Ice::AsyncResultPtr& __result)
{
    ::Ice::AsyncResult::__check(__result, this, __Streamer__BitmapProvider__getBitmap_name);
    ::Streamer::bytes __ret;
    if(!__result->__wait())
    {
        try
        {
            __result->__throwUserException();
        }
        catch(const ::Ice::UserException& __ex)
        {
            throw ::Ice::UnknownUserException(__FILE__, __LINE__, __ex.ice_name());
        }
    }
    ::IceInternal::BasicStream* __is = __result->__getIs();
    __is->startReadEncaps();
    ::std::pair<const ::Ice::Byte*, const ::Ice::Byte*> _____ret;
    __is->read(_____ret);
    ::std::vector< ::Ice::Byte>(_____ret.first, _____ret.second).swap(__ret);
    __is->endReadEncaps();
    return __ret;
}

const ::std::string&
IceProxy::Streamer::BitmapProvider::ice_staticId()
{
    return ::Streamer::BitmapProvider::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::Streamer::BitmapProvider::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::Streamer::BitmapProvider);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::Streamer::BitmapProvider::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::Streamer::BitmapProvider);
}

::IceProxy::Ice::Object*
IceProxy::Streamer::BitmapProvider::__newInstance() const
{
    return new BitmapProvider;
}

void
IceDelegateM::Streamer::BitmapProvider::setCamera(::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __Streamer__BitmapProvider__setCamera_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(ex);
        __os->write(ey);
        __os->write(ez);
        __os->write(dx);
        __os->write(dy);
        __os->write(dz);
        __os->write(ax);
        __os->write(ay);
        __os->write(az);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    if(!__og.is()->b.empty())
    {
        try
        {
            if(!__ok)
            {
                try
                {
                    __og.throwUserException();
                }
                catch(const ::Ice::UserException& __ex)
                {
                    ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                    throw __uue;
                }
            }
            __og.is()->skipEmptyEncaps();
        }
        catch(const ::Ice::LocalException& __ex)
        {
            throw ::IceInternal::LocalExceptionWrapper(__ex, false);
        }
    }
}

::Streamer::bytes
IceDelegateM::Streamer::BitmapProvider::getBitmap(::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __Streamer__BitmapProvider__getBitmap_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(timer);
        __os->write(depthOfField);
        __os->write(transparentColor);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    ::Streamer::bytes __ret;
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        ::IceInternal::BasicStream* __is = __og.is();
        __is->startReadEncaps();
        ::std::pair<const ::Ice::Byte*, const ::Ice::Byte*> _____ret;
        __is->read(_____ret);
        ::std::vector< ::Ice::Byte>(_____ret.first, _____ret.second).swap(__ret);
        __is->endReadEncaps();
        return __ret;
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateD::Streamer::BitmapProvider::setCamera(::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Float ex, ::Ice::Float ey, ::Ice::Float ez, ::Ice::Float dx, ::Ice::Float dy, ::Ice::Float dz, ::Ice::Float ax, ::Ice::Float ay, ::Ice::Float az, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_ex(ex),
            _m_ey(ey),
            _m_ez(ez),
            _m_dx(dx),
            _m_dy(dy),
            _m_dz(dz),
            _m_ax(ax),
            _m_ay(ay),
            _m_az(az)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::Streamer::BitmapProvider* servant = dynamic_cast< ::Streamer::BitmapProvider*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->setCamera(_m_ex, _m_ey, _m_ez, _m_dx, _m_dy, _m_dz, _m_ax, _m_ay, _m_az, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Float _m_ex;
        ::Ice::Float _m_ey;
        ::Ice::Float _m_ez;
        ::Ice::Float _m_dx;
        ::Ice::Float _m_dy;
        ::Ice::Float _m_dz;
        ::Ice::Float _m_ax;
        ::Ice::Float _m_ay;
        ::Ice::Float _m_az;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __Streamer__BitmapProvider__setCamera_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(ex, ey, ez, dx, dy, dz, ax, ay, az, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

::Streamer::bytes
IceDelegateD::Streamer::BitmapProvider::getBitmap(::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Streamer::bytes& __result, ::Ice::Float timer, ::Ice::Float depthOfField, ::Ice::Float transparentColor, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _result(__result),
            _m_timer(timer),
            _m_depthOfField(depthOfField),
            _m_transparentColor(transparentColor)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::Streamer::BitmapProvider* servant = dynamic_cast< ::Streamer::BitmapProvider*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            _result = servant->getBitmap(_m_timer, _m_depthOfField, _m_transparentColor, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Streamer::bytes& _result;
        ::Ice::Float _m_timer;
        ::Ice::Float _m_depthOfField;
        ::Ice::Float _m_transparentColor;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __Streamer__BitmapProvider__getBitmap_name, ::Ice::Normal, __context);
    ::Streamer::bytes __result;
    try
    {
        _DirectI __direct(__result, timer, depthOfField, transparentColor, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
    return __result;
}

::Ice::ObjectPtr
Streamer::BitmapProvider::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __Streamer__BitmapProvider_ids[2] =
{
    "::Ice::Object",
    "::Streamer::BitmapProvider"
};

bool
Streamer::BitmapProvider::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__Streamer__BitmapProvider_ids, __Streamer__BitmapProvider_ids + 2, _s);
}

::std::vector< ::std::string>
Streamer::BitmapProvider::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__Streamer__BitmapProvider_ids[0], &__Streamer__BitmapProvider_ids[2]);
}

const ::std::string&
Streamer::BitmapProvider::ice_id(const ::Ice::Current&) const
{
    return __Streamer__BitmapProvider_ids[1];
}

const ::std::string&
Streamer::BitmapProvider::ice_staticId()
{
    return __Streamer__BitmapProvider_ids[1];
}

::Ice::DispatchStatus
Streamer::BitmapProvider::___setCamera(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Float ex;
    ::Ice::Float ey;
    ::Ice::Float ez;
    ::Ice::Float dx;
    ::Ice::Float dy;
    ::Ice::Float dz;
    ::Ice::Float ax;
    ::Ice::Float ay;
    ::Ice::Float az;
    __is->read(ex);
    __is->read(ey);
    __is->read(ez);
    __is->read(dx);
    __is->read(dy);
    __is->read(dz);
    __is->read(ax);
    __is->read(ay);
    __is->read(az);
    __is->endReadEncaps();
    setCamera(ex, ey, ez, dx, dy, dz, ax, ay, az, __current);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
Streamer::BitmapProvider::___getBitmap(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Float timer;
    ::Ice::Float depthOfField;
    ::Ice::Float transparentColor;
    __is->read(timer);
    __is->read(depthOfField);
    __is->read(transparentColor);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::Streamer::bytes __ret = getBitmap(timer, depthOfField, transparentColor, __current);
    if(__ret.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        __os->write(&__ret[0], &__ret[0] + __ret.size());
    }
    return ::Ice::DispatchOK;
}

static ::std::string __Streamer__BitmapProvider_all[] =
{
    "getBitmap",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping",
    "setCamera"
};

::Ice::DispatchStatus
Streamer::BitmapProvider::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__Streamer__BitmapProvider_all, __Streamer__BitmapProvider_all + 6, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __Streamer__BitmapProvider_all)
    {
        case 0:
        {
            return ___getBitmap(in, current);
        }
        case 1:
        {
            return ___ice_id(in, current);
        }
        case 2:
        {
            return ___ice_ids(in, current);
        }
        case 3:
        {
            return ___ice_isA(in, current);
        }
        case 4:
        {
            return ___ice_ping(in, current);
        }
        case 5:
        {
            return ___setCamera(in, current);
        }
    }

    assert(false);
    throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
}

void
Streamer::BitmapProvider::__write(::IceInternal::BasicStream* __os) const
{
    __os->writeTypeId(ice_staticId());
    __os->startWriteSlice();
    __os->endWriteSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__write(__os);
#else
    ::Ice::Object::__write(__os);
#endif
}

void
Streamer::BitmapProvider::__read(::IceInternal::BasicStream* __is, bool __rid)
{
    if(__rid)
    {
        ::std::string myId;
        __is->readTypeId(myId);
    }
    __is->startReadSlice();
    __is->endReadSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__read(__is, true);
#else
    ::Ice::Object::__read(__is, true);
#endif
}

// COMPILERFIX: Stream API is not supported with VC++ 6
#if !defined(_MSC_VER) || (_MSC_VER >= 1300)
void
Streamer::BitmapProvider::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type Streamer::BitmapProvider was not generated with stream support";
    throw ex;
}

void
Streamer::BitmapProvider::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type Streamer::BitmapProvider was not generated with stream support";
    throw ex;
}
#endif

void 
Streamer::__patch__BitmapProviderPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::Streamer::BitmapProviderPtr* p = static_cast< ::Streamer::BitmapProviderPtr*>(__addr);
    assert(p);
    *p = ::Streamer::BitmapProviderPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::Streamer::BitmapProvider::ice_staticId(), v->ice_id());
    }
}
