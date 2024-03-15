import React, { Component } from "react";
import TopFiveODs from './DomIntRouteChart';
import GeographycalChart from './GeographycalChart';
import BoughtDropdownChart from './BoughtDropout'
import DemographyAgeChart from './DemographyAgeChart';
import Cards from './Cards';
import APIServices from '../../../../API/apiservices';

const apiServices = new APIServices();

class Graphs extends Component {
  constructor(props) {
    super(props);
    this.state = {
    }
  }

  renderTop5OD = () => {
    return (
      <TopFiveODs
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props} />
    )
  }

  renderGeographycalChart = () => {
    return (
      <GeographycalChart
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        ODId={this.props.ODId}
        {...this.props}
      />
    )
  }

  renderSalesAnalysis = () => {
    return (
      <BoughtDropdownChart
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        cityId={this.props.cityId}
        {...this.props} />
    )
  }

  renderDemographyAgeChart = () => {
    return (
      <DemographyAgeChart
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
          {this.renderTop5OD()}
          {this.renderGeographycalChart()}
        </div>

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderSalesAnalysis()}
          {this.renderDemographyAgeChart()}
        </div>

        {this.renderCards()}

      </div>

    )
  }

}

export default Graphs;