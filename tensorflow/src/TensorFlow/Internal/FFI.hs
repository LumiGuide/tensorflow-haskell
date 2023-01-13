-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.Internal.FFI
    ( TensorFlowException(..)
    , Raw.Session
    , Raw.SessionOptions
    , withSession
    , withSessionFromSavedModel
    , SessionAction
    , Raw.Graph
    , importGraphDef
    , run
    , TensorData(..)
    , setSessionConfig
    , setSessionTarget
    , getAllOpList
      -- * Internal helper.
    , useProtoAsVoidPtrLen
    )
    where

import Control.Concurrent.Async (Async, async, cancel, waitCatch)
import Control.Concurrent.MVar (MVar, modifyMVarMasked_, newMVar, takeMVar)
import Control.Monad (when)
import Control.Monad.Catch (MonadMask, Exception, throwM, bracket, finally, mask_)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Data.Bits (Bits, toIntegralSized)
import Data.Int (Int64)
import Data.Maybe (fromMaybe)
import Data.Typeable (Typeable)
import Data.Word (Word8)
import Foreign (Ptr, FunPtr, nullPtr, castPtr)
import Foreign.C (CInt)
import Foreign.C.String (CString)
import Foreign.ForeignPtr (newForeignPtr, newForeignPtr_, withForeignPtr)
import Foreign.Marshal.Alloc (free)
import Foreign.Marshal.Array (withArrayLen, peekArray, mallocArray, copyArray)
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.Encoding.Error as T
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import Data.ProtoLens (Message, encodeMessage)
import Proto.Tensorflow.Core.Framework.Graph (GraphDef)
import Proto.Tensorflow.Core.Framework.Types (DataType(..))
import Proto.Tensorflow.Core.Protobuf.Config (ConfigProto)

import qualified TensorFlow.Internal.Raw as Raw

data TensorFlowException = TensorFlowException Raw.Code T.Text
    deriving (Show, Eq, Typeable)

instance Exception TensorFlowException

-- | All of the data needed to represent a tensor.
data TensorData = TensorData
    { tensorDataDimensions :: [Int64]
    , tensorDataType       :: !DataType
    , tensorDataBytes      :: !(S.Vector Word8)
    }
  deriving (Show, Eq)

-- | The action can spawn concurrent tasks which will be canceled before
-- withSession returns.
type SessionAction m a = (IO () -> IO ()) -> Raw.Session -> Raw.Graph -> m a

-- | Runs the given action after creating a session with options
-- populated by the given optionSetter.
withSession :: (MonadIO m, MonadMask m)
            => (Raw.SessionOptions -> IO ())
            -> SessionAction m a
            -> m a
withSession = withSession_ Raw.newSession

withSessionFromSavedModel :: (MonadIO m, MonadMask m)
                          => B.ByteString
                          -- ^ exportDir
                          -> [B.ByteString]
                          -- ^ Tags.
                          -> (Raw.SessionOptions -> IO ())
                          -- ^ optionSetter
                          -> SessionAction m a
                          -> m a
withSessionFromSavedModel exportDir tags =
    withSession_ $ \graph options status ->
    B.useAsCString exportDir $ \cExportDir ->
    withStringArrayLen tags $ \nTags cTags ->
        Raw.loadSessionFromSavedModel options
                                      runOptions
                                      cExportDir
                                      cTags (safeConvert nTags)
                                      graph
                                      metaGraphDef
                                      status
  where
    runOptions = nullPtr
    metaGraphDef = nullPtr

withSession_ :: (MonadIO m, MonadMask m)
             => (Raw.Graph -> Raw.SessionOptions -> Raw.Status -> IO Raw.Session)
             -- ^ mkSession
             -> (Raw.SessionOptions -> IO ())
             -- ^ optionSetter
             -> SessionAction m a
             -> m a
withSession_ mkSession optionSetter action = do
    drain <- liftIO $ newMVar []
    let cleanup s =
        -- Closes the session to nudge the pending run calls to fail and exit.
            finally (checkStatus (Raw.closeSession s)) $ do
                runners <- takeMVar drain
                -- Collects all runners before deleting the session.
                mapM_ shutDownRunner runners
                checkStatus (Raw.deleteSession s)
    let bracketIO x y = bracket (liftIO x) (liftIO . y)
    bracketIO Raw.newGraph Raw.deleteGraph $ \graph ->
        bracketIO Raw.newSessionOptions Raw.deleteSessionOptions $ \options -> do
            bracketIO
                (optionSetter options >> checkStatus (mkSession graph options))
                cleanup
                (\session -> action (asyncCollector drain) session graph)

asyncCollector :: MVar [Async ()] -> IO () -> IO ()
asyncCollector drain runner = modifyMVarMasked_ drain launchAndRecord
    where
      launchAndRecord restRunners = (: restRunners) <$> async runner

shutDownRunner :: Async () -> IO ()
shutDownRunner r = do
    cancel r
    -- TODO(gnezdo): manage exceptions better than print.
    either print (const (return ())) =<< waitCatch r

importGraphDef :: Raw.Graph -> GraphDef -> IO ()
importGraphDef graph pb =
    useProtoAsBuffer pb $ \buffer ->
        checkStatus $ Raw.importGraphDef graph buffer importGraphDefOptions
  where
    importGraphDefOptions = nullPtr

run :: Raw.Session
    -> Raw.Graph
    -> [(B.ByteString, TensorData)] -- ^ Inputs.
    -> [B.ByteString]               -- ^ Outputs.
    -> [B.ByteString]               -- ^ Target operations.
    -> IO [TensorData]
run session graph inputNamesData outputNames targetNames = do
    -- Use mask to avoid leaking input tensors before they are passed to 'run'
    -- and output tensors before they are passed to 'createTensorData'.
    mask_ $
        -- Inputs.
        mapM (resolveOutput . fst) inputNamesData >>= \inputs ->
        withArrayLen inputs $ \nInputs cInputs ->
        mapM (createRawTensor . snd) inputNamesData >>= \inputTensors ->
        withArrayLen inputTensors $ \_ cInputTensors ->
        -- Outputs.
        mapM resolveOutput outputNames >>= \outputs ->
        withArrayLen outputs $ \nOutputs cOutputs ->
        -- outputTensors is an array of null Tensor pointers that will be filled
        -- by the call to Raw.run.
        withArrayLen (replicate nOutputs nullTensor) $ \_ cOutputTensors ->
        -- Target operations.
        mapM resolveOperation targetNames >>= \targets ->
        withArrayLen targets $ \nTargets cTargets -> do
            checkStatus $ Raw.run
                session
                nullPtr -- RunOptions proto.
                cInputs  cInputTensors  (safeConvert nInputs)
                cOutputs cOutputTensors (safeConvert nOutputs)
                cTargets                (safeConvert nTargets)
                nullPtr -- RunMetadata.
            mapM_ Raw.deleteTensor inputTensors
            outTensors <- peekArray nOutputs cOutputTensors
            mapM createTensorData outTensors
  where
    resolveOutput :: B.ByteString -> IO Raw.Output
    resolveOutput name = do
        let (opName, idx) = parseName name
        op <- resolveOperation opName
        pure $ Raw.Output op idx

    resolveOperation :: B.ByteString -> IO Raw.Operation
    resolveOperation name = do
        op <- B.useAsCString name $ Raw.graphOperationByName graph
        case op of
          Raw.Operation ptr | ptr == nullPtr -> throwM exception
          _ -> pure op
      where
        exception =
            let msg = "Operation not found in graph: " <> (T.pack $ show name)
            in TensorFlowException Raw.TF_INVALID_ARGUMENT msg

    parseName :: B.ByteString -> (B.ByteString, CInt)
    parseName name =
        case break (== ':') (C.unpack name) of
          (name, ':':idxStr) | [(idx, "" :: String)] <- read idxStr
            -> (C.pack name, idx)
          _ -> (name, 0)

    nullTensor = Raw.Tensor nullPtr


-- Internal.


-- | Same as 'fromIntegral', but throws an error if conversion is "lossy".
safeConvert ::
    forall a b. (Show a, Show b, Bits a, Bits b, Integral a, Integral b)
    => a -> b
safeConvert x =
    fromMaybe
    (error ("Failed to convert " ++ show x ++ ", got " ++
            show (fromIntegral x :: b)))
    (toIntegralSized x)


-- | Use a list of ByteString as a list of CString.
withStringList :: [B.ByteString] -> ([CString] -> IO a) -> IO a
withStringList strings fn = go strings []
  where
    go [] cs = fn (reverse cs)
    -- TODO(fmayle): Is it worth using unsafeAsCString here?
    go (x:xs) cs = B.useAsCString x $ \c -> go xs (c:cs)


-- | Use a list of ByteString as an array of CString.
withStringArrayLen :: [B.ByteString] -> (Int -> Ptr CString -> IO a) -> IO a
withStringArrayLen xs fn = withStringList xs (`withArrayLen` fn)


-- | Create a Raw.Tensor from a TensorData.
createRawTensor :: TensorData -> IO Raw.Tensor
createRawTensor (TensorData dims dt byteVec) =
    withArrayLen (map safeConvert dims) $ \cdimsLen cdims -> do
        let len = S.length byteVec
        dest <- mallocArray len
        S.unsafeWith byteVec $ \x -> copyArray dest x len
        Raw.newTensor (toEnum $ fromEnum dt)
                      cdims (safeConvert cdimsLen)
                      (castPtr dest) (safeConvert len)
                      tensorDeallocFunPtr nullPtr

{-# NOINLINE tensorDeallocFunPtr #-}
tensorDeallocFunPtr :: FunPtr Raw.TensorDeallocFn
tensorDeallocFunPtr = unsafePerformIO $ Raw.wrapTensorDealloc $ \x _ _ -> free x

-- | Create a TensorData from a Raw.Tensor.
--
-- Takes ownership of the Raw.Tensor.
-- TODO: Currently, it just makes a copy of the Tensor (and then deletes it),
-- since the raw pointer may refer to storage inside a mutable TensorFlow
-- variable.  We should avoid that copy when it's not needed; for example,
-- by making TensorData wrap an IOVector, and changing the code that uses it.
createTensorData :: Raw.Tensor -> IO TensorData
createTensorData t = do
    -- Read dimensions.
    numDims <- Raw.numDims t
    dims <- mapM (Raw.dim t) [0..numDims-1]
    -- Read type.
    dtype <- toEnum . fromEnum <$> Raw.tensorType t
    -- Read data.
    len <- safeConvert <$> Raw.tensorByteSize t
    bytes <- castPtr <$> Raw.tensorData t :: IO (Ptr Word8)
    fp <- newForeignPtr_ bytes
    -- Make an explicit copy of the raw data, since it might point
    -- to a mutable variable's memory.
    v <- S.freeze (M.unsafeFromForeignPtr0 fp len)
    Raw.deleteTensor t
    return $ TensorData (map safeConvert dims) dtype v

-- | Runs the given action which does FFI calls updating a provided
-- status object. If the status is not OK it is thrown as
-- TensorFlowException.
checkStatus :: (Raw.Status -> IO a) -> IO a
checkStatus fn =
    bracket Raw.newStatus Raw.deleteStatus $ \status -> do
        result <- fn status
        code <- Raw.getCode status
        when (code /= Raw.TF_OK) $ do
            msg <- T.decodeUtf8With T.lenientDecode <$>
                   (Raw.message status >>= B.packCString)
            throwM $ TensorFlowException code msg
        return result

setSessionConfig :: ConfigProto -> Raw.SessionOptions -> IO ()
setSessionConfig pb opt =
    useProtoAsVoidPtrLen pb $ \ptr len ->
        checkStatus (Raw.setConfig opt ptr len)

setSessionTarget :: B.ByteString -> Raw.SessionOptions -> IO ()
setSessionTarget target = B.useAsCString target . Raw.setTarget

-- | Serializes the given msg and provides it as (ptr,len) argument
-- to the given action.
useProtoAsVoidPtrLen :: (Message msg, Integral c, Show c, Bits c) =>
                        msg -> (Ptr b -> c -> IO a) -> IO a
useProtoAsVoidPtrLen msg f = B.useAsCStringLen (encodeMessage msg) $
        \(bytes, len) -> f (castPtr bytes) (safeConvert len)

-- | Serializes the given msg and provides it as BufferPtr argument
-- to the given action.
useProtoAsBuffer :: (Message msg) =>
                    msg -> (Raw.BufferPtr -> IO a) -> IO a
useProtoAsBuffer msg f =
    B.useAsCStringLen (encodeMessage msg) $ \(bytes, len) ->
        bracket (Raw.newBufferFromString (castPtr bytes) (safeConvert len))
                Raw.deleteBuffer
                f

-- | Returns the serialized OpList of all OpDefs defined in this
-- address space.
getAllOpList :: IO B.ByteString
getAllOpList =
    bracket checkCall Raw.deleteBuffer $ \buffer ->
        -- Makes a copy because it is more reliable than eviscerating
        -- Buffer to steal its memory (including custom deallocator).
        B.packCStringLen =<< (,)
            <$> (castPtr <$> Raw.getBufferData buffer)
            <*> (safeConvert <$> Raw.getBufferLength buffer)
    where
      checkCall = do
          p <- Raw.getAllOpList
          when (p == nullPtr) (throwM exception)
          return p
      exception = TensorFlowException
                Raw.TF_UNKNOWN "GetAllOpList failure, check logs"
