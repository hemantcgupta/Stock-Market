import React, { Component } from 'react';
import loader from '../loader.png';

class Loader extends Component {
  render() {
    return (
      <div className="loader" id="loaderImage">
        <img src={loader} className="App-loader" alt="logo" />
      </div>
    );
  }
}

export default Loader;
