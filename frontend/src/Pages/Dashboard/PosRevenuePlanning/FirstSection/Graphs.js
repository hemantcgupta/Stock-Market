import React, { Component } from "react";
import TopODAgentPRP from './TopODAgentPRP';
import RegionPRP from './RegionPRP';
import SalesAnalysisPRP from './SalesAnalysisPRP'
import TargetRational from './TargetRational';
import Cards from './Cards';
import APIServices from '../../../../API/apiservices';

const apiServices = new APIServices();

class Graphs extends Component {
  constructor(props) {
    super(props);
    this.state = {
    }
  }

  renderTopODAgentPRP = () => {
    return (
      <TopODAgentPRP
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props} />
    )
  }

  renderRegionPRP = () => {
    return (
      <RegionPRP
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props}
      />
    )
  }

  renderSalesAnalysis = () => {
    return (
      <SalesAnalysisPRP
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props} />
    )
  }

  renderTargetRational = () => {
    return (
      <TargetRational
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props}
      />
    )
  }

  renderCards = () => {
    return (
      <Cards
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props}
      />
    )
  }

  render() {
    return (
      <div className="row graphs">

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderTopODAgentPRP()}
          {this.renderRegionPRP()}
        </div>

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderSalesAnalysis()}
          {this.renderTargetRational()}
        </div>

        {this.renderCards()}

      </div>

    )
  }

}

export default Graphs;