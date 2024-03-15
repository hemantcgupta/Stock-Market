import React, { Component } from "react";
import RevenueBudget from './RevenueBudget';
import RaskBudget from './RaskBudget';
import YeildBudget from './YeildBudget';
import LoadFactorBudget from './LoadFactorBudget';
import CardsRRP from './CardsRRP';
import APIServices from '../../../../API/apiservices';

const apiServices = new APIServices();

class GraphsRRP extends Component {
  constructor(props) {
    super(props);
    this.state = {
    }
  }

  renderRevenueBudget = () => {
    return (
      <RevenueBudget
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

  renderRaskBudget = () => {
    return (
      <RaskBudget
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

  renderYeildBudget = () => {
    return (
      <YeildBudget
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

  renderLoadFactorBudget = () => {
    return (
      <LoadFactorBudget
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

  renderCardsRRP = () => {
    return (
      <CardsRRP
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
          {this.renderRevenueBudget()}
          {this.renderRaskBudget()}
        </div>

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderYeildBudget()}
          {this.renderLoadFactorBudget()}
        </div>

        {this.renderCardsRRP()}

      </div>

    )
  }

}

export default GraphsRRP;