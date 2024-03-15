import React from "react"

const spinners = (props) =>
  (<div className="spinnertable" style={{ marginTop: props.removeTopMargin ? '0' : '5%' }}>
    <div className="rect1"></div>
    <div className="rect2"></div>
    <div className="rect3"></div>
    <div className="rect4"></div>
    <div className="rect5"></div>
    <div className="rect6"></div>
  </div>)

export default spinners
