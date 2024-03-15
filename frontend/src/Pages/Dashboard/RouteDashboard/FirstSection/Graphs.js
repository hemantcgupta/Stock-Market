import React, { Component } from "react";
import Revenue from './Revenue';
import Rask from './Rask';
import Yeild from './Yeild';
import LoadFactor from './LoadFactor';
import Cards from './Cards';
import APIServices from '../../../../API/apiservices';

const apiServices = new APIServices();

class Graphs extends Component {
  constructor(props) {
    super(props);
    this.state = {
    }
  }

  renderRevenue = () => {
    return (
      <Revenue
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderRask = () => {
    return (
      <Rask
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderYeild = () => {
    return (
      <Yeild
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderLoadFactor = () => {
    return (
      <LoadFactor
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderCards = () => {
    return (
      <Cards
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  render() {
    return (
      <div className="row graphs">

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderRevenue()}
          {this.renderRask()}
        </div>

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderYeild()}
          {this.renderLoadFactor()}
        </div>

        {this.renderCards()}

      </div>

    )
  }

}

export default Graphs;