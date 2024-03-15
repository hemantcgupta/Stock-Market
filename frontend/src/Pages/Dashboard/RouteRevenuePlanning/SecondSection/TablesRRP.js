import React, { Component } from "react";
import RegionBudget from './RegionBudget';
import CabinBudget from './CabinBudget';
import TopBottomRoutesBudget from './TopBottomRoutesBudget';
import '../RouteRevenuePlanning.scss';

class TablesRRP extends Component {

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <RegionBudget
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            {...this.props}
          />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <CabinBudget
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            posDashboard={false}
            {...this.props}
          />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <TopBottomRoutesBudget
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            {...this.props}
          />
        </div>
      </div>
    )
  }
}
export default TablesRRP;